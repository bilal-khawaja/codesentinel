from typing import Dict, List

from ..graphdb.db import GraphDocument, GraphNode, GraphRelationship, GraphSource
from .chunker import MODULE_SYMBOL
from .extractor import CodeInfo, Symbol


def build_graph_documents(
    info: CodeInfo,
    chunk_list: List[dict],
    file_name: str,
    summaries: Dict[str, str],
) -> List[GraphDocument]:
    """One GraphDocument per chunk, in the same shape the LLM extraction path produces.
    chunk_list is the output of create_relation_between_chunks (needs the chunk ids).

    Node ids are scoped by file ("a.py::Class.method") so same-named symbols in
    different files stay separate entities. Only DEFINES (containment) is known from one
    file alone. The raw calls, references, bases and imports are stored as node properties,
    and linker.py turns them into CALLS/USES/INHERITS/IMPORTS edges once it can see the
    other files too."""
    by_qualname = {s.qualname: s for s in info.symbols}
    nodes = {s.qualname: _symbol_node(s, info, file_name, summaries) for s in info.symbols}
    module_node = _module_node(info, file_name, summaries)
    nodes[MODULE_SYMBOL] = module_node

    # Module node lives in the module chunk, or in the first chunk when there is none.
    module_chunk = next(
        (i for i, row in enumerate(chunk_list) if MODULE_SYMBOL in row["chunk_doc"].metadata["symbols"]), 0
    )

    graph_documents = []
    for i, row in enumerate(chunk_list):
        meta = row["chunk_doc"].metadata
        doc_nodes, relationships = [], []

        for qualname in meta["symbols"]:
            if qualname == MODULE_SYMBOL:
                continue
            symbol = by_qualname[qualname]
            doc_nodes.append(nodes[qualname])
            # a symbol split over several chunks gets its DEFINES edge from the chunk it starts in
            if meta["start_line"] <= symbol.start_line <= meta["end_line"]:
                owner = nodes[symbol.parent] if symbol.parent else module_node
                relationships.append(GraphRelationship(owner, nodes[qualname], "DEFINES"))

        if i == module_chunk:
            doc_nodes.append(module_node)

        source = GraphSource(metadata={"combined_chunk_ids": [row["chunk_id"]]})
        graph_documents.append(GraphDocument(nodes=doc_nodes, relationships=relationships, source=source))
    return graph_documents


def _symbol_id(file_name: str, qualname: str) -> str:
    return f"{file_name}::{qualname}"


def _describe(signature: str, text: str) -> str:
    return f"{signature} - {text}" if text else signature


def _symbol_node(symbol: Symbol, info: CodeInfo, file_name: str, summaries: Dict[str, str]) -> GraphNode:
    summary = symbol.docstring or summaries.get(symbol.qualname, "")
    return GraphNode(
        id=_symbol_id(file_name, symbol.qualname),
        type=symbol.kind,
        properties={
            "name": symbol.name,
            "qualname": symbol.qualname,
            "parent": symbol.parent,
            "language": info.language,
            "file": file_name,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "signature": symbol.signature,
            "summary": summary,
            "description": _describe(symbol.signature, summary),
            "bases": symbol.bases,
            "calls": symbol.calls,
            "refs": symbol.refs,
            "locals": symbol.locals,
            "sourceType": "code",
        },
    )


def _module_node(info: CodeInfo, file_name: str, summaries: Dict[str, str]) -> GraphNode:
    summary = info.module_docstring or summaries.get(MODULE_SYMBOL, "")
    return GraphNode(
        id=file_name,
        type="Module",
        properties={
            "name": file_name,
            "language": info.language,
            "file": file_name,
            "start_line": 1,
            "end_line": info.line_count,
            "summary": summary,
            "description": _describe(f"module {file_name}", summary),
            "imports": info.imports,
            "sourceType": "code",
        },
    )
