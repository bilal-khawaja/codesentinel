"""Writes an AnalysisReport into Neo4j.

Graph shape
    (:Finding {id, kind, category, cwe, severity, confidence, title, explanation, scenario, fix,
               file, symbol, start_line, end_line, evidence, detector, status, run_id, model, embedding})
    (:Finding)-[:AFFECTS]->(:__Entity__)   the function, class or module the finding is in
    (:Finding)-[:FOUND_IN]->(:Chunk)       the chunk holding the offending line
    Module entity gets security_summary, risk_level, findings_count, analyzed_hash, analyzed_at.

`Finding` is its own label, not __Entity__, so findings never take part in community detection,
entity embedding or the entity full-text index (those exclude it). Saving a file's analysis
replaces the file's earlier findings. Dismissed rule findings are kept with status "dismissed"
and the reason, so the audit trail shows what the LLM judged a false positive."""
import logging
import uuid
from typing import List, Optional

from ..graphdb.db import execute_graph_query, get_value_from_env, load_embedding_model
from .analyzer import AnalysisReport
from .models import Finding

CREATE_CONSTRAINT = "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE"
CREATE_FILE_INDEX = "CREATE INDEX IF NOT EXISTS FOR (f:Finding) ON (f.file)"

DELETE_FILE_FINDINGS = "MATCH (f:Finding {file: $file}) DETACH DELETE f"

WRITE_FINDINGS = """
UNWIND $rows AS row
MERGE (f:Finding {id: row.id})
SET f += row.props, f.created_at = timestamp()
"""

WRITE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (f:Finding {id: row.id})
CALL db.create.setNodeVectorProperty(f, "embedding", row.embedding)
"""

LINK_ENTITIES = """
UNWIND $rows AS row
MATCH (f:Finding {id: row.id})
MATCH (e:__Entity__ {id: row.entity_id})
MERGE (f)-[:AFFECTS]->(e)
"""

LINK_CHUNKS = """
UNWIND $rows AS row
MATCH (f:Finding {id: row.id})
CALL {
  WITH row
  MATCH (c:Chunk {fileName: $file})
  WHERE c.start_line <= row.start_line AND row.start_line <= c.end_line
  RETURN c ORDER BY c.end_line - c.start_line LIMIT 1
}
MERGE (f)-[:FOUND_IN]->(c)
"""

WRITE_FILE_SUMMARY = """
MATCH (m:__Entity__ {id: $file, type: 'Module'})
SET m.security_summary = $summary, m.risk_level = $risk_level, m.findings_count = $count,
    m.analyzed_hash = $content_hash, m.analyzed_at = timestamp(), m.analysis_model = $model
"""


def save_analysis(
    graph,
    report: AnalysisReport,
    model: str,
    content_hash: str,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    run_id: Optional[str] = None,
) -> dict:
    """Replace the file's findings with this report. Returns what was written."""
    run_id = run_id or uuid.uuid4().hex[:12]
    findings = [(f, "open", "") for f in report.findings]
    findings += [(d.finding, "dismissed", d.reason) for d in report.dismissed]

    execute_graph_query(graph, CREATE_CONSTRAINT)
    execute_graph_query(graph, CREATE_FILE_INDEX)
    execute_graph_query(graph, DELETE_FILE_FINDINGS, {"file": report.file})

    if findings:
        rows = [_row(f, status, reason, run_id, model) for f, status, reason in findings]
        execute_graph_query(graph, WRITE_FINDINGS, {"rows": rows})
        embedded = _embeddings(findings, embedding_provider, embedding_model)
        if embedded:
            execute_graph_query(graph, WRITE_EMBEDDINGS, {"rows": embedded})
        execute_graph_query(graph, LINK_ENTITIES, {"rows": rows})
        execute_graph_query(graph, LINK_CHUNKS, {"rows": rows, "file": report.file})

    summary = report.summary
    execute_graph_query(graph, WRITE_FILE_SUMMARY, {
        "file": report.file,
        "summary": summary.text if summary else "",
        "risk_level": summary.risk_level if summary else "none",
        "count": len(report.findings),
        "content_hash": content_hash,
        "model": model,
    })
    logging.info(f"Saved {len(report.findings)} findings ({len(report.dismissed)} dismissed) for {report.file}")
    return {"run_id": run_id, "saved": len(report.findings), "dismissed": len(report.dismissed)}


def _row(finding: Finding, status: str, reason: str, run_id: str, model: str) -> dict:
    props = finding.model_dump()
    props.update(status=status, dismissed_reason=reason, run_id=run_id, model=model)
    return {
        "id": finding.id,
        "props": props,
        "entity_id": f"{finding.file}::{finding.symbol}" if finding.symbol else finding.file,
        "start_line": finding.start_line,
    }


def _embeddings(findings, embedding_provider: str, embedding_model: str) -> List[dict]:
    """Vectors for semantic search over findings (title, explanation, scenario)."""
    if not get_value_from_env("IS_EMBEDDING", "True", "bool"):
        return []
    embeddings, _ = load_embedding_model(embedding_provider, embedding_model)
    return [
        {"id": f.id, "embedding": embeddings.embed_query(
            f"{f.title}. {f.category}. {f.explanation} {f.scenario}")}
        for f, _, _ in findings
    ]
