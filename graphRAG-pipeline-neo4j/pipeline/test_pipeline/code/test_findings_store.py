"""Tests for storing findings (phase 3), against a fake graph that records every query. No Neo4j, no LLM."""
import pytest

from pipeline.code_analysis.analyzer import AnalysisReport, FileSummary
from pipeline.code_analysis.reviewer import DismissedFinding
from pipeline.code_analysis.rules import scan_source
from pipeline.code_analysis.store import save_analysis
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.communities import make_communities
from pipeline.graphdb.graphDB_dataAccess import graphDBdataAccess
from pipeline.indexing_embedding import post_processing
from pipeline.shared import constants

SOURCE = '''import os

API_TOKEN = "sk-live-9f8a7b6c5d4e3f2a1b0c"


def ping(host):
    return os.system("ping " + host)


def run(cmd):
    eval(cmd)
'''


class RecordingGraph:
    _database = "neo4j"

    def __init__(self):
        self.calls = []

    def query(self, query, params=None, session_params=None):
        self.calls.append((" ".join(query.split()), params or {}))
        return []

    def find(self, text):
        return [(q, p) for q, p in self.calls if text in q]


@pytest.fixture(autouse=True)
def no_embedding_calls(monkeypatch):
    monkeypatch.setenv("IS_EMBEDDING", "False")


def make_report(dismiss_first=False):
    info = extract(SOURCE, PYTHON)
    findings = scan_source(SOURCE, "pkg/app.py", info)
    dismissed = [DismissedFinding(findings.pop(0), "value is a placeholder")] if dismiss_first else []
    return AnalysisReport(
        file="pkg/app.py", findings=findings, dismissed=dismissed,
        summary=FileSummary("high", "Command injection in ping."),
    )


def test_save_replaces_the_files_findings_before_writing():
    graph = RecordingGraph()
    save_analysis(graph, make_report(), model="gpt-4.1-mini", content_hash="abc")
    queries = [q for q, _ in graph.calls]
    delete_at = next(i for i, q in enumerate(queries) if "DETACH DELETE f" in q)
    write_at = next(i for i, q in enumerate(queries) if "MERGE (f:Finding" in q)
    assert delete_at < write_at
    assert graph.find("DETACH DELETE f")[0][1] == {"file": "pkg/app.py"}


def test_rows_carry_the_full_finding_and_link_to_the_right_entity():
    graph = RecordingGraph()
    result = save_analysis(graph, make_report(), model="gpt-4.1-mini", content_hash="abc", run_id="run1")
    rows = graph.find("MERGE (f:Finding")[0][1]["rows"]
    assert result == {"run_id": "run1", "saved": len(rows), "dismissed": 0}
    by_rule = {r["props"]["rule_id"]: r for r in rows}

    ping = by_rule["shell-command"]
    assert ping["entity_id"] == "pkg/app.py::ping" and ping["start_line"] == 7
    assert by_rule["hardcoded-secret"]["entity_id"] == "pkg/app.py"          # module-level: the Module entity
    props = ping["props"]
    assert props["status"] == "open" and props["run_id"] == "run1" and props["model"] == "gpt-4.1-mini"
    assert props["severity"] == "high" and props["cwe"] == "CWE-78" and props["file"] == "pkg/app.py"
    assert all(props[k] for k in ("title", "explanation", "scenario", "fix", "evidence"))


def test_secret_values_never_reach_the_database():
    graph = RecordingGraph()
    save_analysis(graph, make_report(), model="gpt-4.1-mini", content_hash="abc")
    assert "9f8a7b6c5d4e3f2a1b0c" not in str(graph.calls)


def test_dismissed_findings_are_kept_with_their_reason():
    graph = RecordingGraph()
    result = save_analysis(graph, make_report(dismiss_first=True), model="m", content_hash="abc")
    rows = graph.find("MERGE (f:Finding")[0][1]["rows"]
    dismissed = [r for r in rows if r["props"]["status"] == "dismissed"]
    assert result["dismissed"] == 1 and len(dismissed) == 1
    assert dismissed[0]["props"]["dismissed_reason"] == "value is a placeholder"


def test_file_summary_goes_on_the_module():
    graph = RecordingGraph()
    save_analysis(graph, make_report(), model="gpt-4.1-mini", content_hash="abc")
    query, params = graph.find("m.security_summary")[0]
    assert "id: $file, type: 'Module'" in query
    assert params["summary"] == "Command injection in ping." and params["risk_level"] == "high"
    assert params["content_hash"] == "abc" and params["count"] == 3


def test_file_without_findings_still_clears_old_ones_and_records_the_summary():
    graph = RecordingGraph()
    report = AnalysisReport(file="ok.py", summary=FileSummary("none", "Nothing detected."))
    save_analysis(graph, report, model="m", content_hash="h")
    assert graph.find("DETACH DELETE f") and graph.find("m.security_summary")
    assert not graph.find("MERGE (f:Finding")


def test_embeddings_are_written_when_enabled(monkeypatch):
    monkeypatch.setenv("IS_EMBEDDING", "True")

    class FakeEmbeddings:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr("pipeline.code_analysis.store.load_embedding_model", lambda provider, model: (FakeEmbeddings(), 3))
    graph = RecordingGraph()
    save_analysis(graph, make_report(), model="m", content_hash="h")
    rows = graph.find("setNodeVectorProperty")[0][1]["rows"]
    assert len(rows) == 3 and rows[0]["embedding"] == [0.1, 0.2, 0.3]


def test_findings_are_excluded_from_communities_embeddings_and_entity_index():
    assert "!Finding" in make_communities.NODE_PROJECTION
    assert "Finding" in post_processing.FILTER_LABELS
    graph = RecordingGraph()
    post_processing.fetch_entities_for_embedding(graph)
    assert "e:Finding" in graph.calls[0][0]
    assert any(i["label"] == "Finding" and i["name"] == "finding_vector" for i in post_processing.VECTOR_INDEXES)


def test_existing_retrieval_queries_do_not_walk_into_findings():
    walks = constants.VECTOR_GRAPH_SEARCH_ENTITY_QUERY
    assert walks.count("&!Finding") == 2 and walks.count("!AFFECTS&!FOUND_IN") == 2
    assert "e2:!Chunk&!Finding" in constants.GRAPH_QUERY
    assert "e2:!Chunk&!Finding" in constants.QUERY_TO_GET_NODES_AND_RELATIONS_OF_A_DOCUMENT


def test_reingesting_a_file_deletes_its_findings_too():
    graph = RecordingGraph()
    graphDBdataAccess(graph).delete_stale_document_content("pkg/app.py")
    assert graph.find("MATCH (f:Finding {file: $fn}) DETACH DELETE f")[0][1] == {"fn": "pkg/app.py"}
