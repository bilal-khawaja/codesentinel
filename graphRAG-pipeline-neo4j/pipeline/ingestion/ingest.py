import hashlib
import logging
import asyncio
from ..graphdb.db import (
    Neo4jCredentials,
    create_graph_database_connection,
    save_graphDocuments_in_neo4j,
    handle_backticks_nodes_relationship_id_type,
    get_chunk_and_graphDocument,
)
from ..document_sources.local_file import get_documents_from_file_by_path
from ..document_sources.web_pages import get_documents_from_web_page
from ..document_sources.wikipedia import get_documents_from_wikipedia
from ..chunking.create_chunks import CreateChunksofDocument
from ..relationships.make_relationships import (
    create_relation_between_chunks,
    create_chunk_embeddings,
    create_chunk_vector_index,
    merge_relationship_between_chunk_and_entites,
)
from ..llm_setup.llm import get_graph_from_llm
from ..graphdb.graphDB_dataAccess import graphDBdataAccess, sourceNode
from ..indexing_embedding.post_processing import create_vector_fulltext_indexes, create_entity_embedding
from ..communities.make_communities import create_communities


def _hash_pages(pages) -> str:
    return hashlib.sha1("".join(p.page_content for p in pages).encode("utf-8")).hexdigest()


def _hash_file(source_path) -> str:
    with open(source_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


async def _resolve_existing_document_async(file_name: str, content_hash: str, credentials: Neo4jCredentials) -> str:
    """Non-blocking check for existing document status."""
    def _check():
        graph = create_graph_database_connection(credentials)
        dao = graphDBdataAccess(graph)
        existing_hash = dao.get_document_content_hash(file_name)
        if existing_hash is None:
            return "new"
        if existing_hash == content_hash:
            return "unchanged"
        logging.info(f"'{file_name}' content changed — deleting stale content before re-ingest.")
        dao.delete_stale_document_content(file_name)
        return "changed"

    return await asyncio.to_thread(_check)

def _skipped_result(file_name) -> dict:
    logging.info(f"Skipping ingest: '{file_name}' content is unchanged since last ingest.")
    return {"fileName": file_name, "status": "skipped", "reason": "unchanged_content"}


async def ingest(
    source_path: str,
    model: str,
    credentials: Neo4jCredentials,
    allowed_nodes: str = None,
    allowed_relationships: str = None,
    chunks_to_combine: int = 5,
    token_chunk_size: int = 500,
    chunk_overlap: int = 20,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    rebuild_communities: bool = True,
) -> dict:
    """Load a local file, extract a graph via LLM, and write it into Neo4j.
    If this exact content was already ingested, this is a no-op. If the file
    exists but its content changed, the old chunks/entities are replaced."""
    file_name = source_path.split("/")[-1].split("\\")[-1]
    content_hash = _hash_file(source_path)
    file_name, pages, _ = await asyncio.to_thread(get_documents_from_file_by_path, source_path, file_name)
    if not pages:
        raise ValueError(f"No content extracted from {source_path}")

    status = await _resolve_existing_document_async(file_name, content_hash, credentials)
    if status == "unchanged":
            return _skipped_result(file_name)

    return await _ingest_pages(
        file_name, pages, content_hash, model, credentials, allowed_nodes, allowed_relationships,
        chunks_to_combine, token_chunk_size, chunk_overlap, embedding_provider, embedding_model,
        rebuild_communities
    )


async def _ingest_pages(
    file_name,
    pages,
    content_hash,
    model,
    credentials,
    allowed_nodes,
    allowed_relationships,
    chunks_to_combine,
    token_chunk_size,
    chunk_overlap,
    embedding_provider,
    embedding_model,
    rebuild_communities: bool = True
) -> dict:
    graph = create_graph_database_connection(credentials)
    graphDb_data_Access = graphDBdataAccess(graph)
    obj_source_node = sourceNode()
    obj_source_node.file_name = file_name
    obj_source_node.model = model
    await asyncio.to_thread(graphDb_data_Access.create_source_node, obj_source_node)

    # Clean text content
    for page in pages:
        page.page_content = page.page_content.replace('"', '').replace("'", '').replace('\n', ' ')

    #Chunking & Chunk Relationship Creation
    chunks = CreateChunksofDocument(pages).split_file_into_chunks(
        token_chunk_size, chunk_overlap, embedding_model=embedding_model
    )
    chunkId_chunkDoc_list = await asyncio.to_thread(
        create_relation_between_chunks, graph, file_name, chunks
    )

    # Concurrent Execution: Chunk Embeddings & Async LLM Graph Extraction
    chunk_emb_task = asyncio.to_thread(
        create_chunk_embeddings, graph, chunkId_chunkDoc_list, file_name, embedding_provider, embedding_model
    )
    llm_extract_task = get_graph_from_llm(
        model, chunkId_chunkDoc_list, allowed_nodes, allowed_relationships, chunks_to_combine
    )

    # Run embedding creation and LLM extraction concurrently
    _, (graph_documents, token_usage) = await asyncio.gather(chunk_emb_task, llm_extract_task)

    # Process and Save Graph Documents
    graph_documents = handle_backticks_nodes_relationship_id_type(graph_documents)
    await asyncio.to_thread(save_graphDocuments_in_neo4j, graph, graph_documents)

    # Link Chunks to Extracted Entities
    chunks_and_graphDocuments_list = get_chunk_and_graphDocument(graph_documents, chunkId_chunkDoc_list)
    await asyncio.to_thread(merge_relationship_between_chunk_and_entites, graph, chunks_and_graphDocuments_list)

    # Post-Processing Tasks Run Concurrently
    post_processing_tasks = [
        asyncio.to_thread(create_chunk_vector_index, graph, embedding_provider, embedding_model),
        asyncio.to_thread(create_vector_fulltext_indexes, credentials, embedding_provider, embedding_model),
        asyncio.to_thread(create_entity_embedding, graph, embedding_provider, embedding_model),
    ]
    await asyncio.gather(*post_processing_tasks)

    # Incremental LightRAG Optimization: Optional / Background Community Re-building
    if rebuild_communities:
        await asyncio.to_thread(
            create_communities,
            credentials.uri, credentials.userName, credentials.password, credentials.database,
            model=model, embedding_provider=embedding_provider, embedding_model=embedding_model,
        )

    # Set Document Hash
    await asyncio.to_thread(graphDb_data_Access.set_document_content_hash, file_name, content_hash)

    logging.info(f"Ingest complete for {file_name}, token usage: {token_usage}")
    return {
        "fileName": file_name,
        "totalChunks": len(chunks),
        "nodeCount": sum(len(gd.nodes) for gd in graph_documents),
        "relationshipCount": sum(len(gd.relationships) for gd in graph_documents),
        "tokenUsage": token_usage,
    }


async def ingest_from_url(
    source_url: str,
    model: str,
    credentials: Neo4jCredentials,
    allowed_nodes: str = None,
    allowed_relationships: str = None,
    chunks_to_combine: int = 5,
    token_chunk_size: int = 500,
    chunk_overlap: int = 20,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    rebuild_communities: bool = True,
) -> dict:
    """Load a URL, extract a graph via LLM, and write it into Neo4j.
    If this exact content was already ingested, this is a no-op. If the content
    changed, the old chunks/entities are replaced."""
    
    # Extract filename from URL
    file_name = source_url.split('/')[-1].strip()
    if not file_name or len(file_name) > 255:
        file_name = "webpage"
    
    # Fetch web page content
    pages = await asyncio.to_thread(get_documents_from_web_page, source_url)
    
    if not pages:
        raise ValueError(f"No content extracted from {source_url}")
    
    content_hash = _hash_pages(pages)
    
    # Check if already ingested
    status = await _resolve_existing_document_async(file_name, content_hash, credentials)
    if status == "unchanged":
        return _skipped_result(file_name)
    
    return await _ingest_pages(
        file_name, pages, content_hash, model, credentials, allowed_nodes, allowed_relationships,
        chunks_to_combine, token_chunk_size, chunk_overlap, embedding_provider, embedding_model,
        rebuild_communities
    )


async def ingest_from_wikipedia(
    query: str,
    model: str,
    credentials: Neo4jCredentials,
    language: str = "en",
    allowed_nodes: str = None,
    allowed_relationships: str = None,
    chunks_to_combine: int = 5,
    token_chunk_size: int = 500,
    chunk_overlap: int = 20,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    rebuild_communities: bool = True,
) -> dict:
    """Load a Wikipedia article, extract a graph via LLM, and write it into Neo4j.
    If this exact content was already ingested, this is a no-op. If the content
    changed, the old chunks/entities are replaced."""
    
    # Fetch Wikipedia content (returns tuple: file_name, pages)
    wiki_file_name, pages = await asyncio.to_thread(get_documents_from_wikipedia, query, language)
    
    if not pages:
        raise ValueError(f"No content extracted from Wikipedia query: {query}")
    
    # Use query as filename with wikipedia prefix
    file_name = f"wikipedia_{query.replace(' ', '_')}"
    content_hash = _hash_pages(pages)
    
    # Check if already ingested
    status = await _resolve_existing_document_async(file_name, content_hash, credentials)
    if status == "unchanged":
        return _skipped_result(file_name)
    
    return await _ingest_pages(
        file_name, pages, content_hash, model, credentials, allowed_nodes, allowed_relationships,
        chunks_to_combine, token_chunk_size, chunk_overlap, embedding_provider, embedding_model,
        rebuild_communities
    )