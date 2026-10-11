import os
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import TransientError
from openai import OpenAI


@dataclass
class Neo4jCredentials:
    uri: str
    userName: str
    password: str
    database: str = "neo4j"


# --- Graph document schema (replacement for langchain's Node/Relationship/GraphDocument) ---

@dataclass
class GraphNode:
    id: str
    type: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphRelationship:
    source: GraphNode
    target: GraphNode
    type: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphSource:
    """Minimal stand-in for the langchain Document attached to a GraphDocument
    (only .metadata is used downstream, for combined_chunk_ids)."""
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphDocument:
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    source: GraphSource


class Neo4jGraphConnection:
    """Thin wrapper over the plain neo4j driver that mimics the small surface
    of langchain_neo4j.Neo4jGraph actually used by this pipeline: .query(...)
    and a ._database attribute."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def query(self, query: str, params: Optional[dict] = None, session_params: Optional[dict] = None):
        database = (session_params or {}).get("database", self._database)
        records, _summary, _keys = self._driver.execute_query(
            query, params or {}, database_=database
        )
        return [record.data() for record in records]

    def close(self):
        self._driver.close()


_connections: Dict[tuple, Neo4jGraphConnection] = {}
_connections_lock = threading.Lock()


def create_graph_database_connection(credentials: Neo4jCredentials) -> Neo4jGraphConnection:
    """One shared connection (driver + pool) per set of credentials. The neo4j driver is
    thread-safe and built to be long-lived; opening a new one per call, as this used to,
    costs a fresh connection pool for every file."""
    key = (credentials.uri, credentials.userName, credentials.password, credentials.database)
    with _connections_lock:
        if key not in _connections:
            _connections[key] = Neo4jGraphConnection(
                uri=credentials.uri,
                username=credentials.userName,
                password=credentials.password,
                database=credentials.database,
            )
        return _connections[key]


def save_graphDocuments_in_neo4j(
        graph: Neo4jGraphConnection, graph_document_list: List[GraphDocument],
        max_retries=3, delay=1)-> None:

        """
            Parametric UNWIND batching & incremental upserting for GraphDocuments into Neo4j.
            Replaces row-by-row queries with 2 unified transactions.
        """
        
        if not graph_document_list:
                logging.info("No GraphDocuments to save.")
                return
        # Deduplicate & prepare parameterized node payload
        nodes_map: Dict[str, Dict[str, Any]] = {}
        for gdoc in graph_document_list:
            for node in gdoc.nodes:
                nid = str(node.id).strip()
                ntype = str(node.type).strip() if node.type else "Entity"
                props = node.properties or {}
                desc = props.get("description", "")

                if nid not in nodes_map:
                    nodes_map[nid] = {
                        "id": nid,
                        "type": ntype,
                        "description": desc,
                        "properties": props
                    }
                else:
                    # In-memory merge for duplicate entity descriptions in the same batch, if desc is not same as existing extend the existing description and add 
                    # the new description to it, separated by a pipe (|) character. This ensures that we don't lose any information from duplicate nodes in the same batch.
                    existing_desc = nodes_map[nid]["description"]
                    if desc and desc not in existing_desc:
                        nodes_map[nid]["description"] = f"{existing_desc} | {desc}" if existing_desc else desc

        # Sort nodes by ID to enforce deterministic lock ordering and prevent deadlocks
        nodes_payload = sorted(list(nodes_map.values()), key=lambda x: x["id"])

        # Prepare parameterized relationship payload
        rels_list: List[Dict[str, Any]] = []
        for gdoc in graph_document_list:
            for rel in gdoc.relationships:
                src_id = str(rel.source.id if hasattr(rel.source, "id") else rel.source).strip()
                src_type = str(rel.source.type if hasattr(rel.source, "type") else "Entity").strip()
                tgt_id = str(rel.target.id if hasattr(rel.target, "id") else rel.target).strip()
                tgt_type = str(rel.target.type if hasattr(rel.target, "type") else "Entity").strip()
                rel_type = str(rel.type).strip() if rel.type else "RELATED"
                props = rel.properties or {}

                rels_list.append({
                    "source_id": src_id,
                    "source_type": src_type,
                    "target_id": tgt_id,
                    "target_type": tgt_type,
                    "type": rel_type,
                    "description": props.get("description", ""),
                    "properties": props
                })

        # Sort relationships by source_id, target_id to maintain lock hierarchy
        rels_payload = sorted(rels_list, key=lambda x: (x["source_id"], x["target_id"]))

        # Parametric Cypher Queries with UNWIND
        nodes_query = """
        UNWIND $nodes AS node
        MERGE (n:`__Entity__` {id: node.id})
        ON CREATE SET 
            n.type = node.type,
            n.description = node.description,
            n += node.properties,
            n.created_at = timestamp()
        ON MATCH SET 
            n.type = COALESCE(node.type, n.type),
            n.description = CASE 
                WHEN node.description IS NOT NULL AND node.description <> "" AND NOT n.description CONTAINS node.description 
                THEN n.description + " | " + node.description 
                ELSE n.description 
            END,
            n += node.properties,
            n.updated_at = timestamp()
        """

        rels_query = """
        UNWIND $relationships AS rel
        MERGE (s:`__Entity__` {id: rel.source_id})
            ON CREATE SET s.type = rel.source_type
        MERGE (t:`__Entity__` {id: rel.target_id})
            ON CREATE SET t.type = rel.target_type
        MERGE (s)-[r:`RELATED` {type: rel.type}]->(t)
        ON CREATE SET 
            r.description = rel.description,
            r.weight = 1.0,
            r += rel.properties,
            r.created_at = timestamp()
        ON MATCH SET 
            r.description = CASE 
                WHEN rel.description IS NOT NULL AND rel.description <> "" AND NOT r.description CONTAINS rel.description 
                THEN r.description + " | " + rel.description 
                ELSE r.description 
            END,
            r.weight = COALESCE(r.weight, 1.0) + 1.0,
            r += rel.properties,
            r.updated_at = timestamp()
        """

        # Retry loop wrapping the batch queries
        retries = 0
        while retries < max_retries:
            try:
                if nodes_payload:
                    graph.query(nodes_query, {"nodes": nodes_payload})
                if rels_payload:
                    graph.query(rels_query, {"relationships": rels_payload})
                return
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    retries += 1
                    logging.info(f"Deadlock detected during batch write. Retrying {retries}/{max_retries} in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        logging.error("Failed to execute batch ingestion query after maximum retries due to persistent deadlocks.")
        raise RuntimeError("Batch query execution failed after multiple retries due to deadlock.")

def handle_backticks_nodes_relationship_id_type(graph_document_list: List[GraphDocument]):
    for graph_document in graph_document_list:
        cleaned_nodes = []
        for node in graph_document.nodes:
            node_type = str(node.type).strip() if node.type is not None else ""
            node_id = str(node.id).strip() if node.id is not None else ""
            if node_type and node_id:
                node.type = node_type.replace('`', '')
                node.id = node_id
                cleaned_nodes.append(node)
        cleaned_relationships = []
        for rel in graph_document.relationships:
            src = getattr(rel, "source", None)
            tgt = getattr(rel, "target", None)
            rel_type = str(rel.type).strip() if rel.type is not None else ""
            src_type = str(getattr(src, "type", "")).strip() if src is not None else ""
            src_id = str(getattr(src, "id", "")).strip() if src is not None else ""
            tgt_type = str(getattr(tgt, "type", "")).strip() if tgt is not None else ""
            tgt_id = str(getattr(tgt, "id", "")).strip() if tgt is not None else ""
            if rel_type and src_id and src_type and tgt_id and tgt_type:
                rel.type = rel_type.replace('`', '')
                if src is not None:
                    src.type = src_type.replace('`', '')
                    src.id = src_id.replace('`', '')
                if tgt is not None:
                    tgt.type = tgt_type.replace('`', '')
                    tgt.id = tgt_id
                cleaned_relationships.append(rel)
        graph_document.relationships = cleaned_relationships
        graph_document.nodes = cleaned_nodes
    return graph_document_list


def get_chunk_and_graphDocument(graph_document_list, chunkId_chunkDoc_list):
    lst_chunk_chunkId_document = []
    for graph_document in graph_document_list:
        for chunk_id in graph_document.source.metadata['combined_chunk_ids']:
            lst_chunk_chunkId_document.append({'graph_doc': graph_document, 'chunk_id': chunk_id})
    return lst_chunk_chunkId_document


def execute_graph_query(graph, query, params=None, max_retries=3, delay=2):
    retries = 0
    while retries < max_retries:
        try:
            return graph.query(query, params)
        except TransientError as e:
            if "DeadlockDetected" in str(e):
                retries += 1
                logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
    raise RuntimeError("Query execution failed after multiple retries due to deadlock.")


def convert_type(value: str, data_type: type):
    if data_type in (int, "int"):
        return int(value)
    elif data_type in (float, "float"):
        return float(value)
    elif data_type in (bool, "bool"):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ["true", "1", "yes"]
        raise ValueError(f"Cannot convert {value!r} to bool")
    elif data_type in (list, dict, "list", "dict"):
        return json.loads(value)
    elif data_type in (str, "str"):
        return str(value)
    else:
        raise TypeError(f"Unsupported data type for conversion: {data_type}")


def get_value_from_env(key_name: str, default_value: Any = None, data_type: type = str):
    value = os.getenv(key_name, None)
    if value is not None and value != '':
        return convert_type(value, data_type)
    elif default_value is not None:
        return convert_type(default_value, data_type)
    else:
        logging.error(f"Environment variable '{key_name}' not found and no default value provided.")
        return None


# --- Embedding model loading ---

class OpenAIEmbeddingFunction:
    """Small drop-in replacement for langchain_openai.OpenAIEmbeddings exposing
    just the .embed_query(text) method downstream code relies on, backed
    directly by the `openai` SDK."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        api_key = api_key or get_value_from_env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to pipeline/.env (or export it) "
                "so embeddings can be generated."
            )
        self._client = OpenAI(api_key=api_key)

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ") if text else " "
        response = self._client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


def load_embedding_model(embedding_provider: str, embedding_model_name: str):
    model_dimensions = {
        "openai": {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        },
    }

    provider = embedding_provider.lower()
    model = embedding_model_name

    if provider != "openai" or model not in model_dimensions["openai"]:
        raise ValueError(f"Unsupported provider/model: {provider}/{model}")

    dimension = model_dimensions[provider][model]
    embeddings = OpenAIEmbeddingFunction(model=model)

    logging.info(f"Embedding: Using {provider} - {model}, Dimension: {dimension}")
    return embeddings, dimension


class UniversalTokenUsageHandler:
    """Accumulates token usage across pydantic-ai Agent runs. Unlike the
    original langchain BaseCallbackHandler, this is fed explicitly via
    .add(run_usage) after each agent.run(...) call rather than through a
    callback hook (pydantic-ai does not use langchain's callback manager)."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add(self, usage):
        if usage is None:
            return
        self.total_prompt_tokens += getattr(usage, "input_tokens", 0) or 0
        self.total_completion_tokens += getattr(usage, "output_tokens", 0) or 0

    def report(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
