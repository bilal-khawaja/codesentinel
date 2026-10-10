"""
graphrag_pipeline_pydantic_ai: standalone LLM-based knowledge graph ingestion
and retrieval pipeline over Neo4j, built on pydantic-ai (instead of
langchain) for the LLM/embedding layer and the plain `neo4j` driver (instead
of langchain_neo4j) for graph access. Functional 1:1 port of
graphrag_pipeline/.

Public API:
    ingest(source_path, model, credentials, ...) -> dict
        Loads a local file, extracts a graph via LLM, writes it to Neo4j.

    ingest_from_url(source_url, model, credentials, ...) -> dict
        Loads a URL, extracts a graph via LLM, writes it to Neo4j.

    ingest_from_wikipedia(query, model, credentials, ...) -> dict
        Loads a Wikipedia article, extracts a graph via LLM, writes it to Neo4j.

    retrieve(query, mode, credentials, embedding_provider, embedding_model, model, ...) -> dict
        Runs a single retrieval + answer pass over an existing graph.

    Neo4jCredentials(uri, userName, password, database)
        Connection details shared by ingest functions and retrieve().
"""
from .graphdb.db import Neo4jCredentials
from .ingestion.ingest import ingest, ingest_from_url, ingest_from_wikipedia
from .retrievals.retrieve import retrieve

__all__ = ["ingest", "ingest_from_url", "ingest_from_wikipedia", "retrieve", "Neo4jCredentials"]
