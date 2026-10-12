"""Read side of the vulnerability/bug detector: what the chat agent's findings tools return.

Everything the model or a rule wrote about a finding (title, explanation, scenario, fix, evidence,
dismissal reason, file summary) is derived from untrusted source code, so it is wrapped in
<untrusted> markers: the agent must treat it as data and never follow instructions inside it.
Structured fields the pipeline computed itself (id, severity, file, line, counts) are not wrapped."""
import re
from typing import List, Optional

from ..graphdb.db import create_graph_database_connection, load_embedding_model
from .retrieve import get_neo4j_credentials

SEVERITIES = ["critical", "high", "medium", "low", "info"]
STATUSES = ["open", "dismissed", "all"]
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
TOP_IN_SUMMARY = 5
FINDING_VECTOR_INDEX = "finding_vector"

NOT_PROOF = ("No findings means nothing was detected by the rule scan and the LLM review. "
             "It is not proof the code is safe, and files never analyzed have no findings at all.")

_MARKER = re.compile(r"</?\s*untrusted[^>]*>", re.I)

LIST_QUERY = """
MATCH (f:Finding)
WHERE ($file IS NULL OR f.file = $file OR f.file STARTS WITH $file)
  AND ($severity IS NULL OR f.severity = $severity)
  AND ($kind IS NULL OR f.kind = $kind)
  AND ($status = 'all' OR f.status = $status)
RETURN f.id AS id, f.severity AS severity, f.kind AS kind, f.category AS category, f.cwe AS cwe,
       f.confidence AS confidence, f.file AS file, f.symbol AS symbol, f.start_line AS line,
       f.detector AS detector, f.status AS status, f.title AS title
ORDER BY CASE f.severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
         f.file, f.start_line
LIMIT $limit
"""

GET_QUERY = """
MATCH (f:Finding {id: $id})
OPTIONAL MATCH (f)-[:AFFECTS]->(e:__Entity__)
OPTIONAL MATCH (f)-[:FOUND_IN]->(c:Chunk)
RETURN f {.id, .severity, .kind, .category, .cwe, .confidence, .file, .symbol, .start_line, .end_line,
          .detector, .rule_id, .status, .title, .explanation, .scenario, .fix, .evidence,
          .dismissed_reason, .model} AS finding,
       e.id AS affects, e.signature AS signature, c.text AS code
LIMIT 1
"""

SEARCH_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score
WHERE node.status = 'open' OR $include_dismissed
RETURN node.id AS id, node.severity AS severity, node.kind AS kind, node.category AS category,
       node.cwe AS cwe, node.confidence AS confidence, node.file AS file, node.symbol AS symbol,
       node.start_line AS line, node.detector AS detector, node.status AS status,
       node.title AS title, score
ORDER BY score DESC
"""

SUMMARY_QUERY = """
MATCH (m:__Entity__ {type: 'Module'})
WHERE m.analyzed_hash IS NOT NULL AND ($file IS NULL OR m.id = $file OR m.id STARTS WITH $file)
OPTIONAL MATCH (f:Finding {file: m.id, status: 'open'})
WITH m, collect(f.severity) AS severities
RETURN m.id AS file, m.risk_level AS risk_level, m.security_summary AS summary,
       m.analysis_model AS model, size(severities) AS open_findings,
       [s IN ['critical','high','medium','low','info'] | s + ':' + toString(size([x IN severities WHERE x = s]))] AS by_severity
ORDER BY CASE m.risk_level WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END, m.id
LIMIT $limit
"""

UNANALYZED_QUERY = """
MATCH (m:__Entity__ {type: 'Module'})
WHERE m.analyzed_hash IS NULL AND ($file IS NULL OR m.id = $file OR m.id STARTS WITH $file)
RETURN m.id AS file ORDER BY file LIMIT $limit
"""


def untrusted(text) -> str:
    """Wrap text that came from analysed code. Marker look-alikes inside it are removed so
    the text cannot close the wrapper early."""
    text = _MARKER.sub("", str(text if text is not None else "")).strip()
    return f"<untrusted>{text}</untrusted>"


def _limit(limit: int) -> int:
    return max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))


def _clean(value: Optional[str], allowed: List[str], name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    value = value.lower().strip()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return value


def format_row(row: dict) -> str:
    """One finding on one line: computed fields plain, free text wrapped."""
    where = f"{row['file']}:{row['line']}" + (f" in {row['symbol']}" if row.get("symbol") else "")
    cwe = f" {row['cwe']}" if row.get("cwe") else ""
    status = "" if row.get("status") in (None, "open") else f" [{row['status']}]"
    score = f" score={row['score']:.2f}" if row.get("score") is not None else ""
    return (f"{row['id']} | {row['severity']} {row['kind']} ({row['category']}{cwe}) | confidence {row['confidence']} | "
            f"{where} | found by {row['detector']}{status}{score} | {untrusted(row['title'])}")


def list_findings(graph, file: Optional[str] = None, severity: Optional[str] = None, kind: Optional[str] = None,
                  status: str = "open", limit: int = DEFAULT_LIMIT) -> str:
    severity = _clean(severity, SEVERITIES, "severity")
    kind = _clean(kind, ["vulnerability", "bug"], "kind")
    status = _clean(status, STATUSES, "status") or "open"
    limit = _limit(limit)
    rows = graph.query(LIST_QUERY, {"file": file, "severity": severity, "kind": kind, "status": status, "limit": limit})
    if not rows:
        return "No findings match. " + NOT_PROOF
    head = f"{len(rows)} finding(s), most severe first" + (f" (capped at {limit})" if len(rows) == limit else "") + ":"
    return "\n".join([head] + [format_row(r) for r in rows])


def get_finding(graph, finding_id: str) -> str:
    rows = graph.query(GET_QUERY, {"id": finding_id.strip()})
    if not rows:
        return f"No finding with id {finding_id}."
    row, f = rows[0], rows[0]["finding"]
    lines = [
        f"id: {f['id']}", f"severity: {f['severity']}", f"kind: {f['kind']}",
        f"category: {f['category']}" + (f" ({f['cwe']})" if f.get("cwe") else ""),
        f"confidence: {f['confidence']}", f"status: {f['status']}",
        f"location: {f['file']}:{f['start_line']}-{f['end_line']}" + (f" in {f['symbol']}" if f.get("symbol") else ""),
        f"found by: {f['detector']}" + (f" (rule {f['rule_id']})" if f.get("rule_id") else "")
        + (f", reviewed by {f['model']}" if f.get("model") else ""),
        f"title: {untrusted(f['title'])}",
        f"explanation: {untrusted(f['explanation'])}",
        f"scenario: {untrusted(f['scenario'])}",
        f"suggested fix: {untrusted(f['fix'])}",
        f"evidence: {untrusted(f['evidence'])}",
    ]
    if f.get("dismissed_reason"):
        lines.append(f"dismissed because: {untrusted(f['dismissed_reason'])}")
    if row.get("affects"):
        lines.append(f"affects: {row['affects']}" + (f" - {row['signature']}" if row.get("signature") else ""))
    if row.get("code"):
        lines.append("enclosing code:\n" + untrusted(row["code"]))
    return "\n".join(lines)


def search_findings(graph, query: str, limit: int = 10, include_dismissed: bool = False,
                    embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small") -> str:
    limit = _limit(limit)
    embeddings, _ = load_embedding_model(embedding_provider, embedding_model)
    vector = embeddings.embed_query(query)
    # ask for extra neighbours: the status filter runs after the index lookup
    rows = graph.query(SEARCH_QUERY, {"index": FINDING_VECTOR_INDEX, "k": limit * 3,
                                      "vector": vector, "include_dismissed": include_dismissed})[:limit]
    if not rows:
        return "No similar findings. " + NOT_PROOF
    return "\n".join([f"{len(rows)} closest finding(s):"] + [format_row(r) for r in rows])


def security_summary(graph, file: Optional[str] = None, limit: int = DEFAULT_LIMIT) -> str:
    limit = _limit(limit)
    params = {"file": file, "limit": limit}
    analyzed = graph.query(SUMMARY_QUERY, params)
    never = graph.query(UNANALYZED_QUERY, params)
    top = graph.query(LIST_QUERY, {"file": file, "severity": None, "kind": None, "status": "open", "limit": TOP_IN_SUMMARY})
    lines = []
    for r in analyzed:
        lines.append(f"{r['file']} | risk {r['risk_level']} | open {r['open_findings']} ({', '.join(r['by_severity'])}) | "
                     f"{untrusted(r['summary'])}")
    if top:
        lines.append(f"Top {len(top)} open finding(s), most severe first (use get_finding for detail):")
        lines += [format_row(r) for r in top]
    if never:
        lines.append("Never analyzed (no security information, not safe): " + ", ".join(r["file"] for r in never))
    if not lines:
        return "No analyzed files match. " + NOT_PROOF
    return "\n".join(lines + [NOT_PROOF])


# Thin wrappers for the agent: open the connection from the environment like the other tools do.
def _graph():
    return create_graph_database_connection(get_neo4j_credentials())


def list_findings_tool(**kw) -> str:
    return list_findings(_graph(), **kw)


def get_finding_tool(finding_id: str) -> str:
    return get_finding(_graph(), finding_id)


def search_findings_tool(query: str, **kw) -> str:
    return search_findings(_graph(), query, **kw)


def security_summary_tool(**kw) -> str:
    return security_summary(_graph(), **kw)
