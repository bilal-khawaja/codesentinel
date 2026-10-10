import logging
import hashlib
import time
from typing import Any, Dict, List

from ..graphdb.db import load_embedding_model, execute_graph_query, get_value_from_env
from ..document_sources.document import Document

logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')

CREATE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{embedding_property})
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


def merge_relationship_between_chunk_and_entites(graph, graph_documents_chunk_chunk_Id: list):
    batch_data = []
    logging.info("Create HAS_ENTITY relationship between chunks and entities")

    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        for node in graph_doc_chunk_id['graph_doc'].nodes:
            query_data = {
                'chunk_id': graph_doc_chunk_id['chunk_id'],
                'node_type': node.type,
                'node_id': node.id
            }
            batch_data.append(query_data)

    if batch_data:
        unwind_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:Chunk {id: data.chunk_id})
                    MATCH (n:`__Entity__` {id: data.node_id})
                    MERGE (c)-[:HAS_ENTITY]->(n)
                """
        execute_graph_query(graph, unwind_query, params={"batch_data": batch_data})


def create_chunk_embeddings(graph, chunkId_chunkDoc_list, file_name, embedding_provider, embedding_model):
    isEmbedding = get_value_from_env("IS_EMBEDDING", "True", "bool")

    if not isEmbedding:
        logging.info("Embedding is disabled. Skipping embedding creation.")
        return
    
    embeddings, dimension = load_embedding_model(embedding_provider, embedding_model)
    logging.info(f'embedding model:{embeddings} and dimesion:{dimension}')
    data_for_query = []
    logging.info("update embedding and vector index for chunks")
    for row in chunkId_chunkDoc_list:
        if isEmbedding:
            embeddings_arr = embeddings.embed_query(row['chunk_doc'].page_content)

            data_for_query.append({
                "chunkId": row['chunk_id'],
                "embeddings": embeddings_arr
            })

    # check if data_for_query is not empty before executing the query
    if not data_for_query:

        logging.info("No data for embedding creation. Skipping embedding update.")
        return
    
    query_to_create_embedding = """
        UNWIND $data AS row
        MATCH (d:Document {fileName: $fileName})
        MERGE (c:Chunk {id: row.chunkId})
        SET c.embedding = row.embeddings
        MERGE (c)-[:PART_OF]->(d)
    """
    execute_graph_query(graph, query_to_create_embedding, params={"fileName": file_name, "data": data_for_query})

def create_relation_between_chunks(graph, file_name: str, chunks: List[Document]) -> List[Dict[str, Any]]:
    """Ingests Chunks and links sequential metadata (PART_OF, FIRST_CHUNK, NEXT_CHUNK)."""
    logging.info("Creating Chunk nodes and sequential relationships (FIRST_CHUNK, NEXT_CHUNK)...")
    
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    next_relationships = []
    first_chunk_id = None
    offset = 0

    for i, chunk in enumerate(chunks):
        page_content_sha1 = hashlib.sha1(chunk.page_content.encode('utf-8'))
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1

        if i > 0:
            offset += len(chunks[i - 1].page_content)

        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk.page_content,
            "position": position,
            "length": len(chunk.page_content),
            "f_name": file_name,
            "previous_id": previous_chunk_id,
            "content_offset": offset,
            "page_number": chunk.metadata.get('page_number'),
            "start_time": chunk.metadata.get('start_timestamp'),
            "end_time": chunk.metadata.get('end_timestamp')
        }

        batch_data.append(chunk_data)
        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk})

        if i == 0:
            first_chunk_id = current_chunk_id
        else:
            next_relationships.append({
                "previous_chunk_id": previous_chunk_id,
                "current_chunk_id": current_chunk_id
            })

    if not batch_data:
        return []

    # merge chunks and attach to Document
    query_create_chunks = """
    UNWIND $batch_data AS data
    MATCH (d:Document {fileName: data.f_name})
    MERGE (c:Chunk {id: data.id})
    SET c.text = data.pg_content,
        c.position = data.position,
        c.length = data.length,
        c.fileName = data.f_name,
        c.content_offset = data.content_offset,
        c.embedding = data.embedding,
        c.page_number = data.page_number,
        c.start_time = data.start_time,
        c.end_time = data.end_time
    MERGE (c)-[:PART_OF]->(d)
    """
    execute_graph_query(graph, query_create_chunks, params={"batch_data": batch_data})

    # set FIRST_CHUNK relationship
    if first_chunk_id:
        query_first_chunk = """
        MATCH (d:Document {fileName: $f_name})
        MATCH (c:Chunk {id: $first_chunk_id})
        MERGE (d)-[:FIRST_CHUNK]->(c)
        """
        execute_graph_query(graph, query_first_chunk, params={"f_name": file_name, "first_chunk_id": first_chunk_id})

    # connect sequential NEXT_CHUNK relationships
    if next_relationships:
        query_next_chunks = """
        UNWIND $rel_data AS rel
        MATCH (pc:Chunk {id: rel.previous_chunk_id})
        MATCH (c:Chunk {id: rel.current_chunk_id})
        MERGE (pc)-[:NEXT_CHUNK]->(c)
        """
        execute_graph_query(graph, query_next_chunks, params={"rel_data": next_relationships})

    return lst_chunks_including_hash

def create_chunk_vector_index(graph, embedding_provider, embedding_model):
    start_time = time.time()
    try:
        vector_index_query = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE name = 'vector' AND type = 'VECTOR' AND 'Chunk' IN labelsOrTypes AND 'embedding' IN properties RETURN name"
        vector_index = execute_graph_query(graph, vector_index_query)
        if not vector_index:
            _EMBEDDING_FUNCTION, EMBEDDING_DIMENSION = load_embedding_model(embedding_provider, embedding_model)
            query = CREATE_VECTOR_INDEX_QUERY.format(
                index_name="vector",
                node_label="Chunk",
                embedding_property="embedding",
                embedding_dimension=EMBEDDING_DIMENSION,
            )
            execute_graph_query(graph, query)
            logging.info(f"Index created successfully. Time taken: {time.time() - start_time:.2f} seconds")
        else:
            logging.info(f"Index already exist,Skipping creation. Time taken: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        if ("EquivalentSchemaRuleAlreadyExists" in str(e) or "An equivalent index already exists" in str(e)):
            logging.info("Vector index already exists, skipping creation.")
        else:
            raise
