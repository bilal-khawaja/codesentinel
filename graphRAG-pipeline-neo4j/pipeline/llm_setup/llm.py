import asyncio
import logging
import re
import time
from typing import List, Optional

import httpx2

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..graphdb.db import UniversalTokenUsageHandler, get_value_from_env, GraphDocument, GraphNode, GraphRelationship, GraphSource
from ..document_sources.document import Document
from ..shared.constants import ADDITIONAL_INSTRUCTIONS
from .metrics import LLMMetrics


class LLMGraphBuilderException(Exception):
    """Exception raised for custom error in the pipeline."""


# --- Structured output schema for graph extraction (replacement for
# langchain_neo4j.LLMGraphTransformer's internal _Graph schema) ---

class _ExtractedNode(BaseModel):
    id: str = Field(description="Unique identifier of the entity, e.g. its name")
    type: str = Field(description="The entity/category type, e.g. Person, Organization")
    description: Optional[str] = Field(default=None, description="Short description of the entity")


class _ExtractedRelationship(BaseModel):
    source_id: str = Field(description="id of the source node")
    source_type: str = Field(description="type of the source node")
    target_id: str = Field(description="id of the target node")
    target_type: str = Field(description="type of the target node")
    type: str = Field(description="Relationship type, e.g. WORKS_FOR")
    description: Optional[str] = Field(default=None, description="Short description of the relationship")


class ExtractedGraph(BaseModel):
    nodes: List[_ExtractedNode] = Field(default_factory=list)
    relationships: List[_ExtractedRelationship] = Field(default_factory=list)


GRAPH_EXTRACTION_SYSTEM_PROMPT = """

You are analyzing a software project for a code knowledge graph.

Extract entities and relationships that describe the structure,
dependencies, and behavior of the source code.

Focus on:
- source files
- classes
- functions and methods
- imported modules
- external dependencies
- APIs
- databases
- authentication and authorization mechanisms
- sensitive operations
- function calls
- security-relevant code

Create relationships only when they are supported by the source code.

Do not invent entities or relationships that cannot be inferred
from the provided code.

Do not extract ordinary natural-language entities unless they
are relevant to the software structure.
"""



class LLMHandle:
    """Wraps a pydantic-ai model so callers can either get a plain text
    completion (chat/summary use-cases, replacing ChatOpenAI.invoke) or a
    structured pydantic result (graph extraction, replacing
    llm.with_structured_output(...))."""

    def __init__(self, model, model_name: str, usage_tracker: UniversalTokenUsageHandler):
        self.model = model
        self.model_name = model_name
        self.usage_tracker = usage_tracker
        self.metrics = LLMMetrics(model_name)

    # Added defensive handling so token recording failures do not break execution flows
    def _record(self, result, label: str, started: float):
        usage = None
        try:
            if hasattr(result, "usage"):
                # a property in current pydantic-ai, a method in older versions
                usage = result.usage() if callable(result.usage) else result.usage
                self.usage_tracker.add(usage)
        except Exception as usage_err:
            logging.error(f"Error while reporting token usage: {usage_err}")
        self.metrics.record(label, started, time.perf_counter(), usage)

    def complete(self, system_prompt: str, user_prompt: str, label: str = "complete") -> str:
        agent = Agent(self.model, output_type=str, system_prompt=system_prompt)
        started = time.perf_counter()
        result = agent.run_sync(user_prompt)
        self._record(result, label, started)
        return result.output

    async def acomplete(self, system_prompt: str, user_prompt: str, label: str = "complete") -> str:
        agent = Agent(self.model, output_type=str, system_prompt=system_prompt)
        started = time.perf_counter()
        result = await agent.run(user_prompt)
        self._record(result, label, started)
        return result.output

    def run_agent(self, agent, prompt: str, label: str = "agent"):
        """Run an agent from create_query_agent() and record its time, tokens and cost
        (all model calls and tool rounds of the run are counted together)."""
        started = time.perf_counter()
        result = agent.run_sync(prompt)
        self._record(result, label, started)
        return result

    def create_query_agent(self):

        from ..retrievals.retrieve import community_retriever, data_retriever, symbol_lookup, symbol_neighbors

        agent = Agent(
            self.model,
            output_type=str,
            system_prompt=
            """"
            You answer questions using the knowledge graph.

            Use search_communities when the question is broad
            or when you need to identify the relevant area of the graph.

            Use retrieve_data when you need the actual data
            required to answer the question.

            For questions about source code, use find_symbol to get a function or class
            (its code, signature and location), and get_neighbors for structure:
            callers (direction "in", rel_types ["CALLS"]), callees (direction "out"),
            subclasses (direction "in", ["INHERITS"]), users of a variable (direction "in", ["USES"])
            and contents of a class or module (["DEFINES"]).

            Do not invent information that was not returned by the tools.
            """
            )
        
        @agent.tool_plain
        def search_communities(query: str) -> str:

            from ..shared.tracing import trace_tool

            with trace_tool(
                "search_communities", 
                query=query
                ):

                print(f"Searching communities for query: {query}")                
                result = community_retriever(query)

                if not result:
                    print("No communities found.")
                    return "No communities found."
                
                print(f"Found communities: {result}")
                for row in result:
                    print(f"Community ID: {row['community_id']}, Score: {row['score']}")

                return str(result)

        @agent.tool_plain
        def retrieve_data(
            query: str,
            community_ids : list[str] | None = None,
            document_names: list[str] | None = None
            ) -> str:
            from ..shared.tracing import trace_tool

            with trace_tool(
                "retrieve_data",
                query=query,
                community_ids=community_ids,
                document_names=document_names,
            ):
                print("Searching underlying graph data...")

                result = data_retriever(
                    query,
                    community_ids,
                    document_names,
                )

                if not result:
                    print("No data rows returned.")
                    return "No data rows returned."
                
                print(f" Data rows returned: {len(result)}")

                for row in result:
                    print(f"      - {dict(row)}")

                return str(result)

        @agent.tool_plain
        def find_symbol(name: str) -> str:
            from ..shared.tracing import trace_tool

            with trace_tool("find_symbol", name=name):
                result = symbol_lookup(name)
                return str(result) if result else "No symbol found."

        @agent.tool_plain
        def get_neighbors(
            symbol: str,
            rel_types: list[str] | None = None,
            direction: str = "out",
            depth: int = 1,
        ) -> str:
            from ..shared.tracing import trace_tool

            with trace_tool("get_neighbors", symbol=symbol, rel_types=rel_types, direction=direction, depth=depth):
                result = symbol_neighbors(symbol, rel_types, direction, depth)
                return str(result) if result else "No related symbols found."

        return agent

    async def astructured(self, system_prompt: str, user_prompt: str, output_type, label: str = "structured"):
        agent = Agent(self.model, output_type=output_type, system_prompt=system_prompt)
        started = time.perf_counter()
        result = await agent.run(user_prompt)
        self._record(result, label, started)
        return result.output


def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    model = model.upper().replace('.', '_').strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = get_value_from_env(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)

    logging.info("Model: {}".format(env_key))
    callback_handler = UniversalTokenUsageHandler()
    try:
        if "OPENAI" in model:
            model_name, api_key = env_value.split(",")
            http_client = httpx2.AsyncClient(timeout=120.0)
            provider = OpenAIProvider(api_key=api_key, http_client=http_client)
            pai_model = OpenAIChatModel(model_name, provider=provider)

        else:
            model_name, api_endpoint, api_key = env_value.split(",")
            http_client = httpx2.AsyncClient(timeout=120.0)
            provider = OpenAIProvider(base_url=api_endpoint, api_key=api_key, http_client=http_client)
            pai_model = OpenAIChatModel(model_name, provider=provider)

        llm = LLMHandle(pai_model, model_name, callback_handler)
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)

    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name, callback_handler


def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i: i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def _build_extraction_system_prompt(allowedNodes, allowedRelationship, additional_instructions=None):
    prompt = GRAPH_EXTRACTION_SYSTEM_PROMPT + "\n" + ADDITIONAL_INSTRUCTIONS
    if additional_instructions:
        prompt += "\n" + sanitize_additional_instruction(additional_instructions)
    if allowedNodes:
        prompt += (
            "\nOnly use the following node types (do not invent new ones): "
            + ", ".join(allowedNodes)
        )
    if allowedRelationship:
        rel_desc = "; ".join(f"({s})-[{r}]->({t})" for s, r, t in allowedRelationship)
        prompt += (
            "\nOnly use the following relationship patterns (source_type-relationship-target_type): "
            + rel_desc
        )
    return prompt


def _extracted_graph_to_graph_document(extracted: ExtractedGraph, combined_chunk_document, allowedNodes, allowedRelationship) -> GraphDocument:
    allowed_node_set = set(allowedNodes) if allowedNodes else None
    allowed_rel_set = set(allowedRelationship) if allowedRelationship else None

    node_lookup = {}
    nodes = []
    for n in extracted.nodes:
        if allowed_node_set is not None and n.type not in allowed_node_set:
            continue
        node_id = n.id.strip() if n.id else ""
        node_type = n.type.strip() if n.type else ""
        properties = {"description": n.description} if n.description else {}
        node = GraphNode(id=node_id, type=node_type, properties=properties)
        node_lookup[(node_id, node_type)] = node
        nodes.append(node)

    relationships = []

    for r in extracted.relationships:
        if allowed_rel_set is not None and (r.source_type, r.type, r.target_type) not in allowed_rel_set:
            continue
        # Applied string stripping to relationship targets and sources for consistent dictionary matching
        src_id = r.source_id.strip() if r.source_id else ""
        src_type = r.source_type.strip() if r.source_type else ""
        tgt_id = r.target_id.strip() if r.target_id else ""
        tgt_type = r.target_type.strip() if r.target_type else ""

        source = node_lookup.get((src_id, src_type))
        target = node_lookup.get((tgt_id, tgt_type))

        if source is None:
            if allowed_node_set is not None and src_type not in allowed_node_set:
                continue
            source = GraphNode(id=src_id, type=src_type)
            node_lookup[(src_id, src_type)] = source
            nodes.append(source)

        if target is None:
            if allowed_node_set is not None and tgt_type not in allowed_node_set:
                continue
            target = GraphNode(id=tgt_id, type=tgt_type)
            node_lookup[(tgt_id, tgt_type)] = target
            nodes.append(target)

        properties = {"description": r.description} if r.description else {}
        relationships.append(GraphRelationship(source=source, target=target, type=r.type, properties=properties))

    return GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=GraphSource(metadata=combined_chunk_document.metadata),
    )


async def get_graph_document_list(
    llm: LLMHandle, combined_chunk_document_list, allowedNodes, allowedRelationship, 
    callback_handler, additional_instructions=None, max_concurrency: int = 5
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(additional_instructions)
    graph_document_list = []
    token_usage = 0
    try:
        semaphore = asyncio.Semaphore(max_concurrency)
        system_prompt = _build_extraction_system_prompt(allowedNodes, allowedRelationship, additional_instructions)

        # Refactored sequential `for` loop to concurrent `asyncio.gather` for parallel LLM extraction.
        # This reduces graph generation latency linearly based on the number of combined chunks.
        async def _extract_doc(doc: Document) -> GraphDocument:
            async with semaphore:
                        extracted = await llm.astructured(system_prompt, doc.page_content, ExtractedGraph)
                        return _extracted_graph_to_graph_document(extracted, doc, allowedNodes, allowedRelationship)
            
        graph_document_list = await asyncio.gather(
            *(_extract_doc(doc) for doc in combined_chunk_document_list)
        )
        
    except Exception as e:
        logging.error(f"Error in graph transformation: {e}", exc_info=True)
        raise LLMGraphBuilderException(f"Graph transformation failed: {str(e)}")
    finally:
        try:
            if callback_handler:
                usage = callback_handler.report()
                # Added defensive dictionary key lookup for token counts across different handlers
                if isinstance(usage, dict):
                    token_usage = usage.get("total_tokens", usage.get("total_token_count", 0))
        except Exception as usage_err:
            logging.error(f"Error while reporting token usage: {usage_err}")

    return list(graph_document_list), token_usage


async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, chunks_to_combine, additional_instructions=None):
    try:
        llm, model_name, callback_handler = get_llm(model)
        logging.info(f"Using model: {model_name}")

        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
        logging.info(f"Combined {len(combined_chunk_document_list)} chunks")

        if allowedNodes:
            allowed_nodes = [node.strip() for node in allowedNodes.split(',') if node.strip()]
        else:
            allowed_nodes = []
        logging.info(f"Allowed nodes: {allowed_nodes}")

        allowed_relationships = []
        if allowedRelationship:
            items = [item.strip() for item in allowedRelationship.split(',') if item.strip()]
            if len(items) % 3 != 0:
                raise LLMGraphBuilderException("allowedRelationship must be a multiple of 3 (source, relationship, target)")
            for i in range(0, len(items), 3):
                source, relation, target = items[i:i + 3]
                if source not in allowed_nodes or target not in allowed_nodes:
                    raise LLMGraphBuilderException(
                        f"Invalid relationship ({source}, {relation}, {target}): "
                        f"source or target not in allowedNodes"
                    )
                allowed_relationships.append((source, relation, target))
            logging.info(f"Allowed relationships: {allowed_relationships}")
        else:
            logging.info("No allowed relationships provided")

        graph_document_list, token_usage = await get_graph_document_list(
            llm,
            combined_chunk_document_list,
            allowed_nodes,
            allowed_relationships,
            callback_handler,
            additional_instructions,
        )
        logging.info(f"Generated {len(graph_document_list)} graph documents")
        return graph_document_list, token_usage
    except Exception as e:
        logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
        raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")


def sanitize_additional_instruction(instruction: str) -> str:
    """Sanitize additional instruction: neutralize braces and blocked-call patterns, normalize whitespace."""
    logging.info("Sanitizing additional instructions")
    instruction = instruction.replace("{", "[").replace("}", "]")
    injection_patterns = [r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
    for pattern in injection_patterns:
        instruction = re.sub(pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
    instruction = instruction.replace("{", "{{").replace("}", "}}")
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    return instruction
