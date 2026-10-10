# GraphRAG Pipeline (Pydantic AI + Neo4j)

A Python package that ingests content into a Neo4j knowledge graph and answers questions over it with retrieval-augmented generation. It is a library: import it from your own code. There is no server or UI.

## What it does today

Ingestion (`pipeline/ingestion/ingest.py`):

1. **Load**: a local file (PDF via PyMuPDF; `.txt` and other formats via `unstructured`), a web page, or a Wikipedia article.
2. **Chunk**: token-window chunks sized for the embedding model.
3. **Extract**: an LLM extracts entities and relationships from each group of chunks.
4. **Write**: chunks, entities and relationships go into Neo4j. Chunks are linked with `PART_OF`, `FIRST_CHUNK` and `NEXT_CHUNK`, and entities are linked to chunks with `HAS_ENTITY`.
5. **Index**: chunk and entity embeddings, vector indexes, and full-text indexes.
6. **Communities**: Leiden community detection (Neo4j GDS), with LLM-written summaries embedded and indexed.

Unchanged content is skipped, using a content hash. Changed content replaces the old chunks and entities.

Retrieval (`pipeline/retrievals/retrieve.py`): `retrieve()` runs one search-and-answer pass. Modes: `vector`, `fulltext`, `entity_vector`, `graph_vector`, `graph_vector_fulltext` (default), `global_vector`, `graph`. A pydantic-ai query agent (`llm_setup/llm.py`) also exposes `search_communities` and `retrieve_data` tools.

Only text and document data is supported today. Source code is not.

## Planned: source code ingestion

Add Python code as an ingestable source, parsed with tree-sitter instead of treated as prose. [code-graph-rag](https://github.com/vitali87/code-graph-rag) is the inspiration for the parser layout. We are borrowing its pattern, not its code.

| Area | Change |
|---|---|
| New `pipeline/code_parsing/` | Language spec registry, grammar loader, AST extractor, AST-boundary chunker, graph builder. |
| Chunking | One chunk per function, class or method, plus a module chunk. Whitespace is kept, and each chunk has a path/symbol header. |
| Graph | Deterministic nodes (`Module`, `Class`, `Function`) and edges (`DEFINES`, `IMPORTS`, `INHERITS`, `CALLS`) from the AST, with no LLM extraction. Entity IDs are scoped by module path so same-named symbols don't merge. |
| Ingest | `ingest()` routes `.py` files to a code path. The post-processing step is shared with the text path. |
| Communities | Return the real relationship type (`r.type`) in summaries, and add a code-aware prompt. |
| Retrieval | New agent tools: `find_symbol` and `get_neighbors` (callers, callees, inheritance). |
| Later | Repo-wide ingest, cross-file call resolution, more languages. |

`CALLS` edges are resolved within a single file. Ambiguous or external calls are skipped rather than guessed.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [just](https://github.com/casey/just) (optional, for the shortcut commands)
- Docker (for a local Neo4j with APOC and Graph Data Science)
- An OpenAI API key. Embeddings only support OpenAI (`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`).

## Getting started

1. **Install dependencies**

   ```bash
   uv sync
   ```

2. **Configure environment**

   ```bash
   cp pipeline/example.env pipeline/.env
   ```

   Fill in the values:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=testpassword
   NEO4J_DATABASE=neo4j

   # One line per model you plan to use. The key is matched against the model= argument.
   LLM_MODEL_CONFIG_OPENAI_GPT_4_1_MINI="gpt-4.1-mini,sk-..."

   # Used for embeddings.
   OPENAI_API_KEY=sk-...
   ```

   Non-OpenAI models use `deployment_or_model,endpoint,api_key` as the value. See `example.env` for the optional tuning variables.

3. **Start Neo4j**

   ```bash
   just run
   ```

   This runs `uv sync` and `docker compose -f pipeline/docker-compose.yml up -d`. Without `just`, run the compose command directly. The bundled Neo4j uses the password `testpassword`, with the browser on http://localhost:7474 and bolt on port 7687. Wait for the healthcheck to pass (about 30 seconds) before ingesting.

4. **Ingest and query**

   ```python
   import asyncio
   from pipeline import ingest, ingest_from_url, ingest_from_wikipedia, retrieve, Neo4jCredentials

   creds = Neo4jCredentials(
       uri="bolt://localhost:7687",
       userName="neo4j",
       password="testpassword",
       database="neo4j",
   )

   asyncio.run(ingest(
       source_path="/path/to/document.pdf",
       model="openai_gpt_4_1_mini",
       credentials=creds,
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
   ))

   asyncio.run(ingest_from_url("https://example.com/article", model="openai_gpt_4_1_mini", credentials=creds))
   asyncio.run(ingest_from_wikipedia("Amazon (company)", model="openai_gpt_4_1_mini", credentials=creds))

   answer = retrieve(
       query="Who founded Amazon?",
       mode="graph_vector",
       credentials=creds,
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       model="openai_gpt_4_1_mini",
   )
   print(answer["answer"], answer["sources"])
   ```

   Run scripts with `uv run python your_script.py`.

   Optional ingest arguments: `allowed_nodes`, `allowed_relationships`, `chunks_to_combine`, `token_chunk_size`, `chunk_overlap`, `rebuild_communities`.

## Project layout

```text
pipeline/
├── chunking/            token-window chunking
├── communities/         Leiden communities, summaries, embeddings
├── document_sources/    file, web page and Wikipedia loaders
├── graphdb/             Neo4j connection, writers, data access
├── indexing_embedding/  vector and full-text indexes, entity embeddings
├── ingestion/           ingest, ingest_from_url, ingest_from_wikipedia
├── llm_setup/           LLM handles, graph extraction, query agent
├── model_evaluation/    deepeval-based quality checks
├── relationships/       chunk links and chunk embeddings
├── retrievals/          retrieval modes and agent retrievers
├── shared/              constants, prompts, tracing
├── docker-compose.yml   local Neo4j (APOC + GDS)
└── example.env
```

## Testing

There are no tests at the moment. `pipeline/test_pipeline/` was cleared and will be rebuilt alongside the code ingestion work. The `model_evaluation/` checks need a running Neo4j and valid model keys. The `justfile` has commented-out recipes for tests and migrations to enable later.
