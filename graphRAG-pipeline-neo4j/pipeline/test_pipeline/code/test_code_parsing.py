"""Pure tests for code_parsing: no Neo4j, no LLM."""
from pipeline.code_parsing import chunker
from pipeline.code_parsing.chunker import MODULE_SYMBOL, chunk_code
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.graph_builder import build_graph_documents
from pipeline.code_parsing.language_spec import PYTHON, get_spec_for_path
from pipeline.code_parsing.resolver import resolve_links
from pipeline.code_parsing.summarizer import build_summary_inputs

SAMPLE = '''"""Payment helpers."""
import os
from decimal import Decimal

VERSION = "1"


def log(msg):
    print(msg)


class Base:
    pass


class Gateway(Base):
    """Talks to the bank."""

    retries = 3

    def __init__(self):
        self.client = None

    def charge(self, amount: Decimal) -> bool:
        log("charging")
        return self.verify(amount)

    @staticmethod
    def verify(amount):
        """Check the amount."""
        return amount > 0


def main():
    Gateway().charge(1)
    log("done")
    print(VERSION)
'''


def _rows(chunks):
    return [{"chunk_id": f"id{i}", "chunk_doc": c} for i, c in enumerate(chunks)]


def _entities(*files):
    """Entity dicts as the linker reads them back from Neo4j, for (file_name, source) pairs."""
    entities = []
    for file_name, source in files:
        info = extract(source, PYTHON)
        docs = build_graph_documents(info, _rows(chunk_code(source, info, file_name)), file_name, {})
        for doc in docs:
            entities += [{"id": n.id, "type": n.type, **n.properties} for n in doc.nodes]
    return entities


def _edges(*files):
    return resolve_links(_entities(*files))


def test_spec_lookup():
    assert get_spec_for_path("a/b/x.py") is PYTHON
    assert get_spec_for_path("notes.txt") is None


def test_extract_symbols_and_docstrings():
    info = extract(SAMPLE, PYTHON)
    symbols = {s.qualname: s for s in info.symbols}
    assert set(symbols) == {
        "VERSION", "log", "Base", "Gateway", "Gateway.retries", "Gateway.__init__", "Gateway.client",
        "Gateway.charge", "Gateway.verify", "main",
    }
    assert symbols["Gateway.charge"].parent == "Gateway"
    assert symbols["Gateway.charge"].signature == "def charge(self, amount: Decimal) -> bool"
    assert symbols["Gateway"].docstring == "Talks to the bank."
    assert symbols["Gateway.verify"].start_line < symbols["Gateway.verify"].end_line  # decorator included
    assert symbols["VERSION"].kind == symbols["Gateway.client"].kind == "Variable"  # module and instance vars
    assert info.module_docstring == "Payment helpers."


def test_extract_keeps_raw_references():
    info = extract(SAMPLE, PYTHON)
    symbols = {s.qualname: s for s in info.symbols}
    assert info.imports == ["os|os|", "Decimal|decimal|Decimal"]
    assert symbols["Gateway"].bases == ["Base"]
    assert "self.verify" in symbols["Gateway.charge"].calls
    assert set(symbols["main"].calls) == {"Gateway", "Gateway.charge", "log", "print"}  # chained call kept, instantiation too
    assert "VERSION" in symbols["main"].refs
    assert "msg" not in symbols["log"].refs and "msg" in symbols["log"].locals  # parameters are locals, not references


def test_nested_function_calls_are_not_credited_to_outer():
    source = "def helper():\n    pass\n\n\ndef outer():\n    def inner():\n        helper()\n    inner()\n"
    info = extract(source, PYTHON)
    symbols = {s.qualname: s for s in info.symbols}
    assert symbols["outer.inner"].parent == "outer"
    assert symbols["outer.inner"].calls == ["helper"]
    assert symbols["outer"].calls == ["inner"]


def test_chunks_have_unique_headers_and_keep_whitespace():
    info = extract(SAMPLE, PYTHON)
    chunks = chunk_code(SAMPLE, info, "pay.py")
    assert [c.metadata["symbol"] for c in chunks] == [MODULE_SYMBOL, "log", "Base", "Gateway", "main"]
    assert chunks[0].page_content.startswith("# pay.py | <module> | L")
    assert len({c.page_content for c in chunks}) == len(chunks)
    gateway = next(c for c in chunks if c.metadata["symbol"] == "Gateway")
    assert set(gateway.metadata["symbols"]) >= {"Gateway", "Gateway.charge", "Gateway.verify", "Gateway.retries", "Gateway.client"}
    assert "    def charge" in gateway.page_content
    assert "VERSION" in chunks[0].metadata["symbols"]  # module-level variable lives in the module chunk


def test_same_body_in_two_files_gets_different_chunks():
    source = "def f():\n    return 1\n"
    info = extract(source, PYTHON)
    a = chunk_code(source, info, "a.py")[0].page_content
    b = chunk_code(source, info, "b/a.py")[0].page_content
    assert a != b


def test_large_class_split_per_method(monkeypatch):
    monkeypatch.setattr(chunker, "LARGE_CLASS_LINES", 3)
    info = extract(SAMPLE, PYTHON)
    symbols = [c.metadata["symbol"] for c in chunk_code(SAMPLE, info, "pay.py")]
    assert "Gateway.charge" in symbols and "Gateway.verify" in symbols and "Gateway" in symbols


def test_oversized_chunk_split_by_lines_defines_written_once(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHUNK_LINES", 2)
    info = extract(SAMPLE, PYTHON)
    chunks = chunk_code(SAMPLE, info, "pay.py")
    assert [c.metadata["part"] for c in chunks if c.metadata["symbol"] == "main"] == [0, 1]
    docs = build_graph_documents(info, _rows(chunks), "pay.py", {})
    defines = [(r.source.id, r.target.id) for d in docs for r in d.relationships if r.type == "DEFINES"]
    assert len(defines) == len(set(defines))


def test_graph_documents_scoped_ids_and_defines():
    info = extract(SAMPLE, PYTHON)
    chunks = chunk_code(SAMPLE, info, "pay.py")
    docs = build_graph_documents(info, _rows(chunks), "pay.py", {})
    nodes = {n.id: n for d in docs for n in d.nodes}
    assert "pay.py::Gateway.charge" in nodes and "pay.py" in nodes
    assert nodes["pay.py::Gateway.charge"].properties["description"].startswith("def charge(")
    assert nodes["pay.py::Gateway"].properties["summary"] == "Talks to the bank."
    assert nodes["pay.py"].properties["imports"] == ["os|os|", "Decimal|decimal|Decimal"]

    rels = {(r.source.id, r.type, r.target.id) for d in docs for r in d.relationships}
    assert ("pay.py", "DEFINES", "pay.py::Gateway") in rels
    assert ("pay.py", "DEFINES", "pay.py::VERSION") in rels
    assert ("pay.py::Gateway", "DEFINES", "pay.py::Gateway.charge") in rels
    assert ("pay.py::Gateway", "DEFINES", "pay.py::Gateway.client") in rels
    assert [d.source.metadata["combined_chunk_ids"] for d in docs] == [[f"id{i}"] for i in range(len(chunks))]


def test_same_file_links():
    edges = _edges(("pay.py", SAMPLE))
    assert ("pay.py::Gateway", "INHERITS", "pay.py::Base") in edges
    assert ("pay.py::Gateway.charge", "CALLS", "pay.py::Gateway.verify") in edges  # self.method
    assert ("pay.py::Gateway.charge", "CALLS", "pay.py::log") in edges
    assert ("pay.py::main", "CALLS", "pay.py::Gateway") in edges                   # instantiation
    assert ("pay.py::main", "CALLS", "pay.py::Gateway.charge") in edges            # Gateway().charge()
    assert ("pay.py::main", "USES", "pay.py::VERSION") in edges
    assert not any(target.endswith("print") for _, _, target in edges)             # builtins stay unlinked


def test_local_variable_shadowing_is_not_linked():
    source = "def log(m):\n    pass\n\n\ndef run(log):\n    log('x')\n\n\ndef run2():\n    log = make()\n    log('x')\n"
    edges = _edges(("a.py", source))
    assert not any(src in ("a.py::run", "a.py::run2") and kind == "CALLS" for src, kind, _ in edges)


def test_nested_function_call_edges():
    source = "def helper():\n    pass\n\n\ndef outer():\n    def inner():\n        helper()\n    inner()\n"
    edges = _edges(("a.py", source))
    assert ("a.py::outer.inner", "CALLS", "a.py::helper") in edges
    assert ("a.py::outer", "CALLS", "a.py::outer.inner") in edges
    assert ("a.py::outer", "CALLS", "a.py::helper") not in edges


def test_instance_variable_use():
    source = "class A:\n    def __init__(self):\n        self.client = 1\n\n    def go(self):\n        return self.client.get()\n"
    edges = _edges(("a.py", source))
    assert ("a.py::A.go", "USES", "a.py::A.client") in edges


def test_same_names_in_different_files_stay_separate():
    source = "def main():\n    helper()\n\n\ndef helper():\n    pass\n"
    entities = _entities(("a/x.py", source), ("b/x.py", source))
    assert len({e["id"] for e in entities}) == len(entities)
    edges = resolve_links(entities)
    assert ("a/x.py::main", "CALLS", "a/x.py::helper") in edges
    assert ("b/x.py::main", "CALLS", "b/x.py::helper") in edges
    assert ("a/x.py::main", "CALLS", "b/x.py::helper") not in edges


def test_cross_file_calls_imports_and_inheritance():
    models = "class Base:\n    def save(self):\n        pass\n"
    utils = "def helper():\n    pass\n"
    app = (
        "import utils\n"
        "from models import Base\n"
        "from .utils import helper as h\n\n\n"
        "class Child(Base):\n"
        "    def run(self):\n"
        "        self.save()\n"
        "        utils.helper()\n"
        "        h()\n"
    )
    edges = _edges(("pkg/models.py", models), ("pkg/utils.py", utils), ("pkg/app.py", app))
    assert ("pkg/app.py::Child", "INHERITS", "pkg/models.py::Base") in edges
    assert ("pkg/app.py::Child.run", "CALLS", "pkg/models.py::Base.save") in edges   # inherited, different file
    assert ("pkg/app.py::Child.run", "CALLS", "pkg/utils.py::helper") in edges      # module.function and aliased import
    assert ("pkg/app.py", "IMPORTS", "pkg/utils.py") in edges
    assert ("pkg/app.py", "IMPORTS", "pkg/models.py") in edges


def test_ambiguous_module_is_not_guessed():
    app = "from x import f\n\n\ndef go():\n    f()\n"
    edges = _edges(("a/x.py", "def f():\n    pass\n"), ("b/x.py", "def f():\n    pass\n"), ("app.py", app))
    assert not any(src == "app.py::go" for src, _, _ in edges)


def test_links_do_not_depend_on_ingest_order():
    caller = "from lib import f\n\n\ndef go():\n    f()\n"
    lib = "def f():\n    pass\n"
    first = _edges(("app.py", caller), ("lib.py", lib))
    second = _edges(("lib.py", lib), ("app.py", caller))
    assert first == second and ("app.py::go", "CALLS", "lib.py::f") in first


def test_summaries_only_for_undocumented_functions_and_classes():
    info = extract(SAMPLE, PYTHON)
    inputs = build_summary_inputs(info)
    assert "log" in inputs and "Gateway.charge" in inputs
    assert "Gateway" not in inputs and "Gateway.verify" not in inputs  # have docstrings
    assert "VERSION" not in inputs and MODULE_SYMBOL not in inputs     # variables and documented module

    docs = build_graph_documents(info, _rows(chunk_code(SAMPLE, info, "pay.py")), "pay.py", {"log": "Prints a message."})
    log_node = next(n for d in docs for n in d.nodes if n.id == "pay.py::log")
    assert log_node.properties["description"] == "def log(msg) - Prints a message."


def test_parses_every_project_file_with_correct_lines():
    """Regression: tree-sitter's node.start_point crashed the interpreter on larger files,
    so line numbers come from byte offsets. Real files catch what tiny snippets never did."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    files = [p for p in root.rglob("*.py") if "test_pipeline" not in p.parts]
    assert files
    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        for symbol in extract(source, PYTHON).symbols:
            assert 1 <= symbol.start_line <= symbol.end_line <= len(lines), (path.name, symbol.qualname)
            if symbol.kind != "Variable":
                assert lines[symbol.start_line - 1].lstrip().startswith(("def ", "class ", "async def ", "@")), (path.name, symbol.qualname)
