from functools import lru_cache
from importlib import import_module

from tree_sitter import Language, Parser


@lru_cache(maxsize=None)
def get_parser(language_name: str) -> Parser:
    """Build a tree-sitter parser once per language. Grammar comes from the
    `tree_sitter_<language>` package (e.g. tree-sitter-python)."""
    grammar = import_module(f"tree_sitter_{language_name}")
    return Parser(Language(grammar.language()))
