"""Tests of ingest._analyze_if_stale: an unchanged file is re-analyzed when its content or review model changed. Offline."""
import asyncio

import pytest

from pipeline.code_parsing.language_spec import PYTHON
from pipeline.ingestion import ingest as ingest_module


class FakeGraph:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *_):
        return self.rows


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Call _analyze_if_stale with a stored module row; returns (result, whether analysis ran)."""
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")

    def call(stored_rows, hash_now="h1", model_now="gemini_flash_lite"):
        ran = []

        async def fake_run_analysis(*args):
            ran.append(args)
            return {"status": "ok"}

        monkeypatch.setattr(ingest_module, "create_graph_database_connection", lambda creds: FakeGraph(stored_rows))
        monkeypatch.setattr(ingest_module, "_run_analysis", fake_run_analysis)
        result = asyncio.run(ingest_module._analyze_if_stale(
            str(source), "a.py", PYTHON, hash_now, model_now, True, None, "local", "m",
        ))
        return result, bool(ran)

    return call


def test_same_content_and_model_is_not_analyzed_again(run):
    result, ran = run([{"h": "h1", "model": "gemini_flash_lite"}])
    assert result == {"status": "skipped", "reason": "already_analyzed"} and not ran


def test_other_review_model_is_analyzed_again(run):
    _, ran = run([{"h": "h1", "model": "openai_gpt_4_1_mini"}])
    assert ran


def test_other_content_is_analyzed_again(run):
    _, ran = run([{"h": "old", "model": "gemini_flash_lite"}])
    assert ran


def test_file_without_module_node_is_skipped(run):
    result, ran = run([])
    assert result["reason"] == "no_module_node" and not ran
