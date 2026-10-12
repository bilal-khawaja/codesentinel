"""Tests for the analysis wiring in ingest (phase 4). Graph, LLM and storage are faked."""
import asyncio
import threading

import pytest

from pipeline.code_analysis.models import Finding
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.ingestion import ingest as ing
from pipeline.llm_setup.metrics import LLMMetrics

SOURCE = "import os\n\n\ndef run(name):\n    return os.system('x ' + name)\n"


class FakeLLM:
    def __init__(self):
        self.metrics = LLMMetrics("gpt-4.1-mini")

    async def astructured(self, system_prompt, user_prompt, output_type, label=""):
        return output_type() if label.startswith("review") else output_type(summary="Summary.")


class FakeGraph:
    def __init__(self, rows=None):
        self.rows, self.threads = rows or [], []

    def query(self, q, params=None):
        self.threads.append(threading.current_thread())
        return self.rows


@pytest.fixture
def patched(monkeypatch):
    saved = []
    monkeypatch.setattr(ing, "get_llm", lambda model: (FakeLLM(), model, None))
    monkeypatch.setattr(ing, "save_analysis", lambda graph, report, *a: saved.append(report) or
                        {"saved": len(report.findings), "dismissed": len(report.dismissed)})
    return saved


async def test_run_analysis_ok_and_context_fetched_off_loop(patched):
    graph = FakeGraph()
    info = extract(SOURCE, PYTHON)
    out = await ing._run_analysis(graph, SOURCE, "a.py", info, "h", "m", True, "openai", "e")
    assert out["status"] in ("ok", "partial")
    assert out["findings"] >= 1 and out["reviewModel"] == "m"
    assert out["reviewMetrics"] is not None
    assert graph.threads and all(t is not threading.main_thread() for t in graph.threads)
    assert len(patched) == 1


async def test_run_analysis_never_raises(monkeypatch):
    def boom(model):
        raise RuntimeError("no key")
    monkeypatch.setattr(ing, "get_llm", boom)
    out = await ing._run_analysis(FakeGraph(), SOURCE, "a.py", extract(SOURCE, PYTHON), "h", "m", True, "o", "e")
    assert out == {"status": "failed", "error": "no key"}


async def test_run_analysis_survives_storage_failure(monkeypatch):
    monkeypatch.setattr(ing, "get_llm", lambda m: (FakeLLM(), m, None))
    def boom(*a):
        raise RuntimeError("neo4j down")
    monkeypatch.setattr(ing, "save_analysis", boom)
    out = await ing._run_analysis(FakeGraph(), SOURCE, "a.py", extract(SOURCE, PYTHON), "h", "m", True, "o", "e")
    assert out["status"] == "failed"


async def _stale(tmp_path, monkeypatch, rows):
    path = tmp_path / "a.py"
    path.write_text(SOURCE)
    monkeypatch.setattr(ing, "create_graph_database_connection", lambda c: FakeGraph(rows))
    return await ing._analyze_if_stale(str(path), "a.py", PYTHON, "h1", "m", True, None, "o", "e")


async def test_stale_never_analyzed_gets_analyzed(tmp_path, monkeypatch, patched):
    out = await _stale(tmp_path, monkeypatch, [{"h": None}])
    assert out["status"] in ("ok", "partial") and len(patched) == 1


async def test_stale_already_analyzed_skipped(tmp_path, monkeypatch, patched):
    out = await _stale(tmp_path, monkeypatch, [{"h": "h1"}])
    assert out == {"status": "skipped", "reason": "already_analyzed"} and not patched


async def test_stale_without_module_node_skipped(tmp_path, monkeypatch, patched):
    out = await _stale(tmp_path, monkeypatch, [])
    assert out["reason"] == "no_module_node" and not patched


# --- reviewer false positives seen live (phase 4 run on graphDB_dataAccess.py) ---
from pipeline.code_analysis.context import build_units
from pipeline.code_analysis.reviewer import ReviewedFinding, ReviewOutput, build_review_prompt, merge_review
from pipeline.code_analysis.context import SymbolContext
from pipeline.code_analysis.verify import builds_strings, contradiction

SAFE = '''import time
import logging
from pathlib import Path


def mark(conn, name, h):
    query = """MERGE (d:Document {fileName: $fn}) SET d.hash = $hash"""
    time.sleep(1)
    return conn.execute(query, {"fn": name, "hash": h})


def unsafe(conn, name):
    return conn.execute("SELECT * FROM t WHERE n = '" + name + "'")
'''


def _draft(category, title, evidence="x"):
    return ReviewedFinding(kind="vulnerability", category=category, severity="high", confidence="high",
                           title=title, explanation="e", scenario="s", fix="f", start_line=1, end_line=1,
                           evidence=evidence)


def _units():
    info = extract(SAFE, PYTHON)
    return info, {u.symbol: u for u in build_units(SAFE, info, "a.py")}


def test_prompt_lists_file_imports_and_defined_names():
    info, units = _units()
    prompt = build_review_prompt(units["mark"], [], SymbolContext())
    assert "imports:" in prompt and "time" in prompt and "logging" in prompt and "Path" in prompt
    assert "defined at top level:" in prompt and "unsafe" in prompt


def test_fixed_query_text_is_not_injection():
    _, units = _units()
    assert not builds_strings(units["mark"])
    assert contradiction(_draft("cypher-injection", "Cypher injection"), units["mark"])
    assert contradiction(_draft("sql-injection", "SQL injection"), units["mark"])


def test_built_query_text_is_still_injection():
    _, units = _units()
    assert builds_strings(units["unsafe"])
    assert contradiction(_draft("sql-injection", "SQL injection"), units["unsafe"]) is None


def test_missing_import_contradicted_by_file_imports():
    _, units = _units()
    reason = contradiction(_draft("missing-import", "Missing import of required modules time and logging"), units["mark"])
    assert "time" in reason and "logging" in reason
    assert contradiction(_draft("missing-import", "Missing import of requests"), units["mark"]) is None


def test_merge_review_drops_contradicted_findings():
    info, units = _units()
    out = ReviewOutput(findings=[_draft("missing-import", "Missing import of time", "time.sleep(1)")])
    result = merge_review(units["mark"], [], out, info)
    assert not result.findings and "does import" in result.dropped[0]


def test_unparseable_code_is_never_dropped():
    _, units = _units()
    unit = units["unsafe"]
    unit.lines = [(1, "def broken(:"), (2, "    pass")]
    assert builds_strings(unit)


def test_logging_an_fstring_is_not_building_a_query():
    src = '''import logging


def f(conn, name):
    try:
        conn.execute("MERGE (d:D {n: $n})", {"n": name})
    except Exception as e:
        logging.error(f"failed for {name}: {e}")
        raise
'''
    info = extract(src, PYTHON)
    unit = next(u for u in build_units(src, info, "a.py") if u.symbol == "f")
    assert not builds_strings(unit)
    assert contradiction(_draft("cypher-injection", "Cypher injection"), unit)


def test_praise_and_info_findings_are_dropped():
    _, units = _units()
    unit = units["mark"]
    assert contradiction(_draft("good-practice", "Proper use of parameterized Cypher query"), unit)
    assert contradiction(_draft("retry-logic", "Retry logic handles errors safely"), unit)
    info_only = _draft("note", "Something odd").model_copy(update={"severity": "info"})
    assert contradiction(info_only, unit)
    assert contradiction(_draft("none-dereference", "Unchecked None dereference"), unit) is None


def _unit_of(src):
    info = extract(src, PYTHON)
    return next(u for u in build_units(src, info, "a.py") if u.symbol)


def test_command_and_deserialization_findings_checked_against_the_code():
    safe = _unit_of("import json, subprocess\n\ndef f(h):\n    subprocess.run(['ping', h])\n    return json.loads(h)\n")
    assert contradiction(_draft("command-injection", "Command injection"), safe)
    assert contradiction(_draft("unsafe-deserialization", "Unsafe deserialization"), safe)
    risky = _unit_of("import os, pickle\n\ndef f(h):\n    os.system('x ' + h)\n    return pickle.loads(h)\n")
    assert contradiction(_draft("command-injection", "Command injection"), risky) is None
    assert contradiction(_draft("unsafe-deserialization", "Unsafe deserialization"), risky) is None
    aliased = _unit_of("import pickle as pk\n\ndef f(b):\n    return pk.loads(b)\n")
    assert contradiction(_draft("deserialization", "Deserialization"), aliased) is None
    shell = _unit_of("import subprocess\n\ndef f(c):\n    subprocess.run(c, shell=True)\n")
    assert contradiction(_draft("command-injection", "Command injection"), shell) is None
