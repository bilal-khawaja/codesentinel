"""Tests of code_parsing: language detection, symbol extraction, chunking. Offline."""
from pipeline.code_parsing.chunker import MODULE_SYMBOL, chunk_code
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON, get_spec_for_path

SOURCE = '''"""Module doc."""
import os
from a import b as c

LIMIT = 3


class Base:
    pass


class Child(Base):
    def run(self, x):
        """Run it."""
        return helper(x)


def helper(x):
    return os.path.join(x, "y")
'''


def test_spec_is_chosen_by_extension():
    assert get_spec_for_path("pkg/mod.py") is PYTHON
    assert get_spec_for_path("MOD.PY") is PYTHON
    assert get_spec_for_path("notes.txt") is None


def test_extract_finds_symbols_with_kind_parent_and_lines():
    symbols = {s.qualname: s for s in extract(SOURCE, PYTHON).symbols}
    assert set(symbols) == {"LIMIT", "Base", "Child", "Child.run", "helper"}
    assert symbols["Child.run"].kind == "Function" and symbols["Child.run"].parent == "Child"
    assert symbols["Child"].bases == ["Base"]
    assert symbols["LIMIT"].kind == "Variable"
    assert (symbols["helper"].start_line, symbols["helper"].end_line) == (18, 19)


def test_extract_keeps_docstrings_imports_and_calls():
    info = extract(SOURCE, PYTHON)
    symbols = {s.qualname: s for s in info.symbols}
    assert info.module_docstring == "Module doc."
    assert info.imports == ["os|os|", "c|a|b"]  # "alias|module|name"
    assert symbols["Child.run"].docstring == "Run it."
    assert symbols["Child.run"].calls == ["helper"]
    assert symbols["helper"].calls == ["os.path.join"]


def test_chunks_follow_definition_boundaries():
    chunks = chunk_code(SOURCE, extract(SOURCE, PYTHON), "m.py")
    assert [c.metadata["symbol"] for c in chunks] == [MODULE_SYMBOL, "Base", "Child", "helper"]
    assert chunks[0].page_content.startswith("# m.py | <module> |")
    assert chunks[2].metadata["symbols"] == ["Child", "Child.run"]
    assert all(c.metadata["language"] == "python" for c in chunks)


def test_empty_source_has_no_symbols():
    info = extract("", PYTHON)
    assert info.symbols == [] and info.line_count == 0
