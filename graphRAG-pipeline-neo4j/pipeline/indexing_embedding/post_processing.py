import json
import logging
from neo4j import GraphDatabase
from pydantic import BaseModel

from ..graphdb.db import load_embedding_model, execute_graph_query, get_value_from_env
from ..shared.constants import GRAPH_CLEANUP_PROMPT, DEFAULT_LLM_MODEL
from ..llm_setup.llm import get_llm
from ..graphdb.graphDB_dataAccess import graphDBdataAccess

# Constants for Full-Text Indexes
LABELS_QUERY = "CALL db.labels()"
FILTER_LABELS = ["Chunk", "Document", "__Community__", "Finding"]
FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX entities FOR (n{labels_str}) ON EACH [n.id, n.description];"
HYBRID_SEARCH_FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX keyword FOR (n:Chunk) ON EACH [n.text]"
COMMUNITY_INDEX_FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX community_keyword FOR (n:`__Community__`) ON EACH [n.summary]"

# Constants for Vector Indexes
CHUNK_VECTOR_INDEX_NAME = "vector"
ENTITY_VECTOR_INDEX_NAME = "entity_vector"
COMMUNITY_VECTOR_INDEX_NAME = "community_vector"
FINDING_VECTOR_INDEX_NAME = "finding_vector"
VECTOR_EMBEDDING_DEFAULT_DIMENSION = 384

CREATE_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{embedding_property})
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {embedding_dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""

FULLTEXT_INDEXES = [
    {"type": "entities", "query": FULL_TEXT_QUERY},
    {"type": "hybrid", "query": HYBRID_SEARCH_FULL_TEXT_QUERY},
    {"type": "community", "query": COMMUNITY_INDEX_FULL_TEXT_QUERY}
]

VECTOR_INDEXES = [
    {"name": CHUNK_VECTOR_INDEX_NAME, "label": "Chunk", "property": "embedding"},
    {"name": ENTITY_VECTOR_INDEX_NAME, "label": "__Entity__", "property": "embedding"},
    {'name': COMMUNITY_VECTOR_INDEX_NAME, 'label': '__Community__', 'property': 'embedding'},
    {"name": FINDING_VECTOR_INDEX_NAME, "label": "Finding", "property": "embedding"},
]


def get_graphDB_driver(credentials):
    return GraphDatabase.driver(
        credentials.uri,
        auth=(credentials.userName, credentials.password),
        database=credentials.database,
    )


def create_vector_index(session, index_name, node_label, embedding_property, embedding_dimension):
    session.run(f"DROP INDEX {index_name} IF EXISTS;")
    query = CREATE_VECTOR_INDEX_QUERY.format(
        index_name=index_name,
        node_label=node_label,
        embedding_property=embedding_property,
        embedding_dimension=embedding_dimension
    )
    session.run(query)
    logging.info(f"Vector index '{index_name}' created successfully.")


def create_fulltext_index(session, index_type, query):
    drop_query = f"DROP INDEX {index_type} IF EXISTS;"
    if index_type == 'hybrid':
        drop_query = "DROP INDEX keyword IF EXISTS;"
    elif index_type == 'community':
        drop_query = "DROP INDEX community_keyword IF EXISTS;"
    session.run(drop_query)

    if index_type == "entities":
        result = session.run(LABELS_QUERY)
        labels = [record["label"] for record in result if record["label"] not in FILTER_LABELS]
        if labels:
            labels_str = ":" + "|".join([f"`{label}`" for label in labels])
            query = query.format(labels_str=labels_str)
        else:
            logging.info("Full-text index for entities not created as no labels were found.")
            return

    session.run(query)
    logging.info(f"Full-text index for '{index_type}' created successfully.")


def create_vector_fulltext_indexes(credentials, embedding_provider, embedding_model):
    """Creates all configured full-text and vector indexes."""
    logging.info("Starting the process of creating full-text and vector indexes.")

    _, dimension = load_embedding_model(embedding_provider, embedding_model)
    if not dimension:
        dimension = VECTOR_EMBEDDING_DEFAULT_DIMENSION

    driver = None
    try:
        driver = get_graphDB_driver(credentials)
        driver.verify_connectivity()
        logging.info("Database connectivity verified.")

        with driver.session() as session:
            for index_config in FULLTEXT_INDEXES:
                try:
                    create_fulltext_index(session, index_config["type"], index_config["query"])
                except Exception as e:
                    logging.error(f"Failed to create full-text index for type '{index_config['type']}': {e}")

            for index_config in VECTOR_INDEXES:
                try:
                    create_vector_index(session, index_config["name"], index_config["label"], index_config["property"], dimension)
                except Exception as e:
                    logging.error(f"Failed to create vector index '{index_config['name']}': {e}")
    except Exception as e:
        logging.error(f"An error occurred during the index creation process: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logging.info("Driver closed successfully.")

    logging.info("Full-text and vector index creation process completed.")


def create_entity_embedding(graph, embedding_provider, embedding_model):
    rows = fetch_entities_for_embedding(graph)
    for i in range(0, len(rows), 1000):
        update_embeddings(rows[i:i + 1000], graph, embedding_provider, embedding_model)


def fetch_entities_for_embedding(graph):
    query = """
                MATCH (e)
                WHERE NOT (e:Chunk OR e:Document OR e:`__Community__` OR e:Finding) AND e.embedding IS NULL AND e.id IS NOT NULL
                RETURN elementId(e) AS elementId, e.id + " " + coalesce(e.description, "") AS text
                """
    result = execute_graph_query(graph, query)
    return [{"elementId": record["elementId"], "text": record["text"]} for record in result]


def update_embeddings(rows, graph, embedding_provider, embedding_model):
    embeddings, dimension = load_embedding_model(embedding_provider, embedding_model)
    logging.info("update embedding for entities")
    for row in rows:
        row['embedding'] = embeddings.embed_query(row['text'])
    query = """
      UNWIND $rows AS row
      MATCH (e) WHERE elementId(e) = row.elementId
      CALL db.create.setNodeVectorProperty(e, "embedding", row.embedding)
      """
    return execute_graph_query(graph, query, params={'rows': rows})


class _GraphCleanupMapping(BaseModel):
    nodes: dict
    relationships: dict


def graph_schema_consolidation(graph):
    graphDb_data_Access = graphDBdataAccess(graph)
    node_labels, relation_labels = graphDb_data_Access.get_nodelabels_relationships()

    # [OPENAI-DISABLED] graph_cleanup_model = get_value_from_env("GRAPH_CLEANUP_MODEL", 'openai_gpt_4_1_mini')
    graph_cleanup_model = get_value_from_env("GRAPH_CLEANUP_MODEL", DEFAULT_LLM_MODEL)
    llm, _, _ = get_llm(graph_cleanup_model)

    nodes_relations_input = {'nodes': node_labels, 'relationships': relation_labels}
    import asyncio
    mappings_obj = asyncio.run(
        llm.astructured(GRAPH_CLEANUP_PROMPT, json.dumps(nodes_relations_input), _GraphCleanupMapping)
    )
    mappings = {"nodes": mappings_obj.nodes, "relationships": mappings_obj.relationships}
    node_mapping = {old: new for new, old_list in mappings['nodes'].items() for old in old_list if new != old}
    relation_mapping = {old: new for new, old_list in mappings['relationships'].items() for old in old_list if new != old}

    logging.info(f"Node Labels: Total = {len(node_labels)}, Reduced to = {len(set(node_mapping.values()))} (from {len(node_mapping)})")
    logging.info(f"Relationship Types: Total = {len(relation_labels)}, Reduced to = {len(set(relation_mapping.values()))} (from {len(relation_mapping)})")

    for old_label, new_label in node_mapping.items():
        query = f"""
                MATCH (n:`{old_label}`)
                SET n:`{new_label}`
                REMOVE n:`{old_label}`
                """
        execute_graph_query(graph, query)

    for old_label, new_label in relation_mapping.items():
        query = f"""
                MATCH (n)-[r:`{old_label}`]->(m)
                CREATE (n)-[r2:`{new_label}`]->(m)
                DELETE r
                """
        execute_graph_query(graph, query)

    return None
