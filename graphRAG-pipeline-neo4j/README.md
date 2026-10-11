# GraphRAG Pipeline (Pydantic AI + Neo4j)

A Python package that ingests content into a Neo4j knowledge graph and answers questions over it with retrieval-augmented generation. It is a library: import it from your own code. There is no server or UI.

It handles two kinds of source:

- **Documents** (PDF, text, web pages, Wikipedia): chunked, and an LLM extracts entities and relationships.
- **Python source code**: parsed with tree-sitter into a graph of modules, classes, functions and variables, with an LLM reviewer that finds vulnerabilities and bugs.

## What it does today

### Documents

Ingestion (`pipeline/ingestion/ingest.py`):

1. **Load**: a local file (PDF via PyMuPDF; `.txt` and other formats via `unstructured`), a web page, or a Wikipedia article.
2. **Chunk**: token-window chunks sized for the embedding model.
3. **Extract**: an LLM extracts entities and relationships from each group of chunks.
4. **Write**: chunks, entities and relationships go into Neo4j. Chunks are linked with `PART_OF`, `FIRST_CHUNK` and `NEXT_CHUNK`, and entities are linked to chunks with `HAS_ENTITY`.
5. **Index**: chunk and entity embeddings, vector indexes, and full-text indexes.
6. **Communities**: Leiden community detection (Neo4j GDS), with LLM-written summaries embedded and indexed.

Unchanged content is skipped, using a content hash that is checked before any parsing. Changed content replaces the old chunks and entities.

### Source code

`ingest()` sends `.py` files down a code path (`pipeline/code_parsing/`). There is no LLM entity extraction and no text cleaning, and whitespace is kept.

- **Symbols**: modules, classes, functions, methods, nested functions, and variables (module-level, class-level and `self.x`). Ids are scoped by file, for example `pkg/a.py::Class.method`, so same-named symbols in different files never merge.
- **Chunks**: one per function or class (per method for large classes), plus one for module-level code. Each starts with a `# path | symbol | L10-45` header line.
- **Edges**: `DEFINES`, `CALLS`, `USES`, `INHERITS`, `IMPORTS`.
  - Calls follow scope rules: locals shadow file-level names, `self.method()` follows base classes, and `Gateway().charge()` resolves through the class.
  - Calls, references and imports are stored as text on the nodes, and a link step turns them into edges. It reads every code entity in the database, so cross-file links resolve whatever the ingest order. Ambiguous imports are never guessed.
  - Anything that cannot be resolved (builtins, third-party code, values of unknown type) stays on the node as text and creates no edge.
- **Summaries**: undocumented functions, classes and modules get a one-sentence LLM summary (`summarize_code=True`). Docstrings are used as they are.
- **Cross-file linking** needs file names that include the folder, so pass `root_dir=` to `ingest()`. Without it the basename is used.
- **Communities** use a code-aware prompt, and summaries now see the real relationship type.

### Vulnerability and bug detection

`pipeline/code_analysis/` finds vulnerabilities and bugs in Python code.

1. **Rule scan** (`rules.py`, no LLM): 18 checks with exact lines. Command and SQL injection, `eval`/`exec`, unsafe deserialization, hardcoded secrets, weak hashes, disabled TLS verification, world-writable permissions, syntax and indentation errors (`IndentationError`, `TabError`), mutable defaults, bare or swallowed `except`, `assert` misuse, unreachable code, and more.
2. **LLM review** (`reviewer.py`): the model reads each function with numbered lines and its callers and callees from the graph. It confirms or dismisses the rule hits and adds new findings. Code and comments are treated as untrusted data.
3. **Verification** (`verify.py`): line numbers are corrected from the quoted code, invented findings are dropped, and secret values are redacted from every LLM-written field.
4. **Summary** (`analyzer.py`): a file summary, with the risk level computed from the findings and not taken from the LLM.
5. **Storage** (`store.py`): `Finding` nodes with the full record and an embedding, linked by `AFFECTS` to the function, class or module and by `FOUND_IN` to the chunk. The summary and risk level go on the Module node. Dismissed rule findings are kept with their reason. A `finding_vector` index supports semantic search.

`Finding` nodes are kept out of community detection, entity embeddings, the entity full-text index and the existing graph retrieval.

Run it on a file today:

```python
import asyncio
from pipeline.llm_setup.llm import get_llm
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.code_analysis.analyzer import analyze_file

source = open("app.py", encoding="utf-8").read()
llm, _, _ = get_llm("openai_gpt_4_1_mini")
report = asyncio.run(analyze_file(llm, source, "app.py", extract(source, PYTHON)))

for finding in report.findings:
    print(finding.severity, finding.category, finding.symbol, finding.start_line, finding.title)
print(report.summary.risk_level, report.summary.text)
print(llm.metrics.report())
```

To store the result, call `save_analysis(graph, report, model, content_hash)` from `pipeline/code_analysis/store.py`.

### LLM metrics

Every LLM handle records the time, tokens and cost of each call (`pipeline/llm_setup/metrics.py`):

```python
llm.metrics.report()    # one readable line
llm.metrics.summary()   # dict: calls, tokens in/out, USD cost, wall and summed seconds, slowest call
llm.metrics.calls       # one record per call, with a label
```

Prices come from the `genai-prices` database that ships with pydantic-ai. For a model it does not know, set `LLM_PRICE_<MODEL>="<input>,<output>"` in USD per million tokens. Without either, cost is reported as unknown. `llm.run_agent(agent, prompt)` records a query-agent run the same way.

### Retrieval

`retrieve()` (`pipeline/retrievals/retrieve.py`) runs one search-and-answer pass. Modes: `vector`, `fulltext`, `entity_vector`, `graph_vector`, `graph_vector_fulltext` (default), `global_vector`, `graph`.

A pydantic-ai query agent (`llm_setup/llm.py`, `create_query_agent`) exposes these tools:

- `search_communities`, `retrieve_data`: documents and communities.
- `find_symbol`: the code, signature and location of a function, class or variable.
- `get_neighbors`: callers, callees, subclasses, variable users and contents, by relationship type, direction and depth.

## Status and what is left

| Area | Status |
|---|---|
| Document ingestion, retrieval, communities | Done |
| Code ingestion: parsing, chunks, graph, cross-file links, summaries | Done (single file at a time) |
| Agent tools `find_symbol` and `get_neighbors` | Done |
| LLM metrics (time, tokens, cost) | Done |
| Detector phase 1: rule scan | Done |
| Detector phase 2: LLM review, verification, file summary | Done |
| Detector phase 3: storing findings in Neo4j | Done |
| Detector phase 4: wire analysis into `ingest()` | **Not done** |
| Detector phase 5: agent tools for findings | **Not done** |
| Detector phase 6: accuracy evaluation | **Not done** |
| Repo-wide ingest | Not started |
| Languages other than Python | Not started |

**Phase 4, analysis inside `ingest()`**
- New arguments `analyze`, `review_model` and `review_all`.
- Run analysis after the link step, because it needs the call graph.
- Analyze files that were ingested earlier but never analyzed, without re-ingesting them.
- Fetch graph context without blocking the event loop.
- Return an `analysis` block and the review's metrics.
- An analysis failure must not fail the ingest.

**Phase 5, agent tools for findings**
- `list_findings`, `get_finding`, `search_findings` and `security_summary`.
- A reviewer role in the system prompt: every claim cites a finding id and lines, and "no findings" means "nothing detected", not "safe".
- Finding text comes from an LLM that read untrusted code, so tool results mark it as data.
- Optional: `analyze_symbol` to review on demand.

**Phase 6, accuracy**
- A labeled set of vulnerable and clean files, to measure precision and recall.
- The model's judgment varies run to run (for example, it sometimes flags a harmless function as low-severity injection), and today only one hand-made sample has been run.

**Repo mode (later)**
- A directory walker and batch hash check (one query for the whole repo, unchanged files skipped before parsing).
- Run the link step once at the end of a batch instead of after every file.
- Delete from the graph files that no longer exist on disk.

**Known limits**
- Python only.
- A variable's type is not tracked, so `x = Gateway(); x.charge()` stays as text and creates no edge.
- Definitions inside module-level `if` or `try` blocks are skipped.
- Review of one function can miss problems that span modules. A call graph and call paths give it context, but this is not true taint analysis.

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

   Optional ingest arguments: `allowed_nodes`, `allowed_relationships`, `chunks_to_combine`, `token_chunk_size`, `chunk_overlap`, `rebuild_communities`, and for code `summarize_code` and `root_dir`.

5. **Ingest source code**

   ```python
   result = asyncio.run(ingest(
       source_path="/path/to/project/pkg/app.py",
       model="openai_gpt_4_1_mini",
       credentials=creds,
       root_dir="/path/to/project",
       rebuild_communities=False,
   ))
   print(result["llmMetrics"])
   ```

   Ingest the files one by one. Calls, inheritance and imports between files link up as each new file arrives.

## Project layout

```text
pipeline/
├── chunking/            token-window chunking
├── code_analysis/       vulnerability and bug detection: rules, reviewer, verification, storage
├── code_parsing/        tree-sitter extractor, chunker, graph builder, resolver, linker, summarizer
├── communities/         Leiden communities, summaries, embeddings
├── document_sources/    file, web page and Wikipedia loaders
├── graphdb/             Neo4j connection, writers, data access
├── indexing_embedding/  vector and full-text indexes, entity embeddings
├── ingestion/           ingest, ingest_from_url, ingest_from_wikipedia
├── llm_setup/           LLM handles, graph extraction, query agent, metrics
├── model_evaluation/    deepeval-based quality checks
├── relationships/       chunk links and chunk embeddings
├── retrievals/          retrieval modes, agent retrievers, code lookups
├── shared/              constants, prompts, tracing
├── test_pipeline/code/  offline tests for the code features
├── docker-compose.yml   local Neo4j (APOC + GDS)
└── example.env
```

## Testing

```bash
uv run pytest pipeline/test_pipeline/code
```

Or `just test-code`. These tests need no Neo4j and no API key. A scripted fake LLM and pydantic-ai's `TestModel` stand in for the model, and a recording fake stands in for the graph. They cover parsing, chunking, edges, cross-file resolution, the rule scan, review verification, metrics and storage queries.

They check our own logic, not the model's judgment. Model quality is what the phase 6 evaluation is for.

The `model_evaluation/` checks (including `code_callers_eval.py`) need a running Neo4j and valid model keys.
