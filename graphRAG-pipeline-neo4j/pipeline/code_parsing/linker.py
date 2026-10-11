import logging

from .resolver import resolve_links

LINK_EDGE_TYPES = ["CALLS", "USES", "INHERITS", "IMPORTS"]

READ_ENTITIES_QUERY = """
MATCH (e:__Entity__ {sourceType: 'code'})
RETURN e.id AS id, e.type AS type, e.file AS file, e.qualname AS qualname, e.parent AS parent,
       e.name AS name, e.bases AS bases, e.calls AS calls, e.refs AS refs,
       e.locals AS locals, e.imports AS imports
"""

DELETE_EDGES_QUERY = """
MATCH (:__Entity__ {sourceType: 'code'})-[r:RELATED]->()
WHERE r.type IN $types
DELETE r
"""

WRITE_EDGES_QUERY = """
UNWIND $edges AS edge
MATCH (a:__Entity__ {id: edge.source})
MATCH (b:__Entity__ {id: edge.target})
MERGE (a)-[r:RELATED {type: edge.type}]->(b)
ON CREATE SET r.weight = 1.0, r.created_at = timestamp()
"""


def link_code_references(graph) -> int:
    """Rebuild every CALLS/USES/INHERITS/IMPORTS edge from the raw references stored on the
    code entities. Reads all code entities, so a call into a file ingested earlier (or later)
    resolves no matter the ingest order, and edges never go stale. Returns the edge count."""
    entities = graph.query(READ_ENTITIES_QUERY)
    edges = resolve_links(entities)
    graph.query(DELETE_EDGES_QUERY, {"types": LINK_EDGE_TYPES})
    if edges:
        rows = [{"source": source, "type": kind, "target": target} for source, kind, target in sorted(edges)]
        graph.query(WRITE_EDGES_QUERY, {"edges": rows})
    logging.info(f"Linked {len(edges)} code references")
    return len(edges)
