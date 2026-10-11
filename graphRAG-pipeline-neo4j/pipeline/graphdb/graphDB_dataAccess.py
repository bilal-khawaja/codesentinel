import logging
import time
from datetime import datetime, timedelta
from neo4j.exceptions import TransientError
from typing import Any, Dict, List, Optional, Tuple, Union
from ..graphdb.db import Neo4jGraphConnection


# use Optional types for safer type-checking
class sourceNode:
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    file_source: Optional[str] = None
    status: Optional[str] = None
    url: Optional[str] = None
    gcsBucket: Optional[str] = None
    gcsBucketFolder: Optional[str] = None
    gcsProjectId: Optional[str] = None
    awsAccessKeyId: Optional[str] = None
    chunkNodeCount: Optional[int] = None
    chunkRelCount: Optional[int] = None
    entityNodeCount: Optional[int] = None
    entityEntityRelCount: Optional[int] = None
    communityNodeCount: Optional[int] = None
    communityRelCount: Optional[int] = None
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None
    model: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processing_time: Optional[Union[float, timedelta]] = None
    error_message: Optional[str] = None
    total_chunks: Optional[int] = None
    language: Optional[str] = None
    is_cancelled: Optional[bool] = None
    processed_chunk: Optional[int] = None
    access_token: Optional[str] = None
    retry_condition: Optional[str] = None
    token_usage: Optional[int] = None
    embedding_model: Optional[str] = None


class graphDBdataAccess:

    def __init__(self, graph: Neo4jGraphConnection):
        self.graph = graph

    def execute_query(self, query, param=None, max_retries=3, delay=2):
        retries = 0
        while retries < max_retries:
            try:
                return self.graph.query(query, param, session_params={"database": self.graph._database})
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    retries += 1
                    logging.info(f"Deadlock detected. Retrying {retries}/{max_retries} in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        logging.error("Failed to execute query after maximum retries due to persistent deadlocks.")
        raise RuntimeError("Query execution failed after multiple retries due to deadlock.")

    def get_document_content_hash(self, file_name: str) -> Optional[str]:
            """Returns the stored content hash for a Document, or None if not found."""
            query = """
            MATCH (d:Document {fileName: $fn})
            RETURN d.contentHash AS contentHash
            """
            try:
                result = self.execute_query(query, {"fn": file_name})
                if result and result[0].get("contentHash"):
                    return result[0]["contentHash"]
                return None
            except Exception as e:
                logging.error(f"Error reading content hash for file '{file_name}': {e}")
                return None

    def set_document_content_hash(self, file_name: str, content_hash: str) -> None:
        """Sets or updates the content hash for a Document node."""
        query = """
        MERGE (d:Document {fileName: $fn}) 
        SET d.contentHash = $hash
        """
        try:
            self.execute_query(query, {"fn": file_name, "hash": content_hash})
        except Exception as e:
            logging.error(f"Error setting content hash for file '{file_name}': {e}")
            raise

    def delete_stale_document_content(self, file_name):
        """Deletes all Chunks for a Document and cleans up orphaned Entities safely."""
        # scoped OPTIONAL MATCH specifically to __Entity__ labels and decoupled DETACH DELETE to avoid multi-node locks/deadlocks
        query = """
        MATCH (d:Document {fileName: $fn})<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
        WITH collect(DISTINCT c) AS chunks, collect(DISTINCT e) AS candidateEntities
        
        FOREACH (c IN chunks | DETACH DELETE c)
        
        WITH candidateEntities
        UNWIND candidateEntities AS e
        WITH DISTINCT e WHERE e IS NOT NULL
        OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(remainingChunk:Chunk)
        WITH e, count(remainingChunk) AS remaining
        WHERE remaining = 0
        DETACH DELETE e
        """
        try:
            self.execute_query(query, {"fn": file_name})
            # findings point at the entities just removed, so they go with them
            self.execute_query("MATCH (f:Finding {file: $fn}) DETACH DELETE f", {"fn": file_name})
            logging.info(f"Successfully removed stale content for document '{file_name}'.")
        except Exception as e:
            logging.error(f"Error deleting stale content for file '{file_name}': {e}")
            raise

    def update_exception_db(self, file_name, exp_msg):
        """Updates the document node status to 'Failed' with error metadata."""
        query = """
        MERGE (d:Document {fileName: $fName}) 
        SET d.status = 'Failed', 
            d.errorMessage = $error_msg,
            d.updatedAt = datetime()
        """
        try:
            self.execute_query(query, {"fName": file_name, "error_msg": exp_msg})
        except Exception as e:
                    logging.error(f"Error updating document status to failed for '{file_name}': {e}")

    def create_source_node(self, obj_source_node: sourceNode):
        try:
            logging.info(f"creating source node if does not exist in database {self.graph._database}")
            self.graph.query("""MERGE(d:Document {fileName :$fn}) SET d.fileSize = $fs, d.fileType = $ft ,
                            d.status = 'New', d.url = $url, d.fileSource = $f_source, d.createdAt = $c_at,
                            d.updatedAt = $u_at, d.processingTime = 0, d.errorMessage = '', d.nodeCount = 0,
                            d.relationshipCount = 0, d.model = $model, d.language = $language,
                            d.is_cancelled = False, d.total_chunks = 0, d.processed_chunk = 0""",
                            {"fn": obj_source_node.file_name, "fs": obj_source_node.file_size,
                             "ft": obj_source_node.file_type, "url": obj_source_node.url,
                             "f_source": obj_source_node.file_source, "c_at": obj_source_node.created_at,
                             "u_at": obj_source_node.created_at, "model": obj_source_node.model,
                             "language": obj_source_node.language},
                            session_params={"database": self.graph._database})
        except Exception as e:
            error_message = str(e)
            logging.info(f"error_message = {error_message}")
            self.update_exception_db(obj_source_node.file_name, error_message)
            raise Exception(error_message)
        
    def create_source_node(self, obj_source_node: sourceNode) -> None:
            """Initializes a new Document source node in the database."""
            query = """
            MERGE (d:Document {fileName: $fn}) 
            SET d.fileSize = $fs, 
                d.fileType = $ft,
                d.status = 'New', 
                d.url = $url, 
                d.fileSource = $f_source, 
                d.createdAt = $c_at,
                d.updatedAt = $u_at, 
                d.processingTime = 0.0, 
                d.errorMessage = '', 
                d.nodeCount = 0,
                d.relationshipCount = 0, 
                d.model = $model, 
                d.language = $language,
                d.is_cancelled = False, 
                d.total_chunks = 0, 
                d.processed_chunk = 0
            """
            c_at = obj_source_node.created_at.isoformat() if obj_source_node.created_at else datetime.now().isoformat()
            u_at = obj_source_node.updated_at.isoformat() if obj_source_node.updated_at else datetime.now().isoformat()

            params = {
                "fn": obj_source_node.file_name,
                "fs": obj_source_node.file_size or 0,
                "ft": obj_source_node.file_type or "",
                "url": obj_source_node.url or "",
                "f_source": obj_source_node.file_source or "",
                "c_at": c_at,
                "u_at": u_at,
                "model": obj_source_node.model or "",
                "language": obj_source_node.language or "en"
            }

            try:
                logging.info(f"Creating source node for file '{obj_source_node.file_name}' on database '{self.graph._database}'.")
                self.execute_query(query, params)
            except Exception as e:
                err_msg = str(e)
                logging.error(f"Failed to create source node for '{obj_source_node.file_name}': {err_msg}")
                self.update_exception_db(obj_source_node.file_name, err_msg)
                raise

    def update_source_node(self, obj_source_node: sourceNode) -> None:
            """Dynamically updates non-null properties on an existing Document node."""
            props: Dict[str, Any] = {}

            if obj_source_node.file_name:
                props['fileName'] = obj_source_node.file_name
            if obj_source_node.status:
                props['status'] = obj_source_node.status
            if obj_source_node.created_at:
                props['createdAt'] = obj_source_node.created_at.isoformat()
            if obj_source_node.updated_at:
                props['updatedAt'] = obj_source_node.updated_at.isoformat()
                
            if obj_source_node.processing_time is not None:
                if isinstance(obj_source_node.processing_time, timedelta):
                    props['processingTime'] = round(obj_source_node.processing_time.total_seconds(), 2)
                elif isinstance(obj_source_node.processing_time, (int, float)):
                    props['processingTime'] = round(float(obj_source_node.processing_time), 2)

            if obj_source_node.node_count is not None:
                props['nodeCount'] = obj_source_node.node_count
            if obj_source_node.relationship_count is not None:
                props['relationshipCount'] = obj_source_node.relationship_count
            if obj_source_node.model:
                props['model'] = obj_source_node.model
            if obj_source_node.total_chunks is not None:
                props['total_chunks'] = obj_source_node.total_chunks
            if obj_source_node.processed_chunk is not None:
                props['processed_chunk'] = obj_source_node.processed_chunk
            if obj_source_node.token_usage is not None:
                props['token_usage'] = obj_source_node.token_usage
            if obj_source_node.embedding_model:
                props['embedding_model'] = obj_source_node.embedding_model

            if not props.get('fileName'):
                raise ValueError("Cannot update Document node without a valid 'fileName'.")

            query = "MERGE (d:Document {fileName: $props.fileName}) SET d += $props"
            try:
                self.execute_query(query, {"props": props})
            except Exception as e:
                err_msg = str(e)
                logging.error(f"Failed to update source node '{obj_source_node.file_name}': {err_msg}")
                self.update_exception_db(obj_source_node.file_name, err_msg)
                raise

    def get_nodelabels_relationships(self):
        """Safely fetches non-system node labels and relationship types."""
        node_query = """
        CALL db.labels() YIELD label
        WHERE NOT label IN ['Document', 'Chunk', '_Bloom_Perspective_', '__Community__', '__Entity__']
        RETURN label ORDER BY label
        """
        relation_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WHERE NOT relationshipType IN ['PART_OF', 'NEXT_CHUNK', 'HAS_ENTITY', '_Bloom_Perspective_', 'FIRST_CHUNK', 'SIMILAR', 'IN_COMMUNITY', 'PARENT_COMMUNITY']
        RETURN relationshipType ORDER BY relationshipType
        """
        try:
            node_result = self.execute_query(node_query)
            relationship_result = self.execute_query(relation_query)

            node_labels = [record["label"] for record in node_result] if node_result else []
            relationship_types = [record["relationshipType"] for record in relationship_result] if relationship_result else []

            return node_labels, relationship_types
        except Exception as e:
            logging.error(f"Error fetching node labels and relationship types: {e}")
            return [], []
