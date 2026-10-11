from dataclasses import dataclass
from typing import List, Tuple

from ..document_sources.document import Document
from .extractor import CodeInfo, Symbol

MODULE_SYMBOL = "<module>"
LARGE_CLASS_LINES = 80    # classes longer than this are chunked per method
MAX_CHUNK_LINES = 120     # chunks longer than this are split by lines

NumberedLines = List[Tuple[int, str]]


@dataclass
class _Piece:
    symbol: str
    lines: NumberedLines


def chunk_code(source: str, info: CodeInfo, file_name: str) -> List[Document]:
    """Split code at definition boundaries: one chunk per top-level function, per class
    (per method when the class is large) and one for module-level code. Every chunk
    starts with a header line so identical bodies in different places stay unique."""
    lines = source.splitlines()

    def numbered(start, end) -> NumberedLines:
        return [(n, lines[n - 1]) for n in range(start, end + 1)]

    pieces = []
    for symbol in info.symbols:
        if symbol.parent is not None or symbol.kind == "Variable":
            continue
        methods = [m for m in info.symbols if m.parent == symbol.qualname and m.kind == "Function"]
        size = symbol.end_line - symbol.start_line + 1
        if methods and size > LARGE_CLASS_LINES:
            for method in methods:
                pieces.append(_Piece(method.qualname, numbered(method.start_line, method.end_line)))
            class_only = lines_outside(numbered(symbol.start_line, symbol.end_line), [(m.start_line, m.end_line) for m in methods])
            pieces.append(_Piece(symbol.qualname, class_only))
        else:
            pieces.append(_Piece(symbol.qualname, numbered(symbol.start_line, symbol.end_line)))

    defined = [(s.start_line, s.end_line) for s in info.symbols if s.parent is None and s.kind != "Variable"]
    module_lines = lines_outside(numbered(1, len(lines)), defined)
    if module_lines:
        pieces.append(_Piece(MODULE_SYMBOL, module_lines))

    pieces.sort(key=lambda piece: piece.lines[0][0])
    return [chunk for piece in pieces for chunk in _to_documents(piece, file_name, info)]


def lines_outside(numbered_lines: NumberedLines, ranges) -> NumberedLines:
    """Lines not inside any (start, end) range, with blank edges trimmed."""
    kept = [(n, text) for n, text in numbered_lines if not any(a <= n <= b for a, b in ranges)]
    while kept and not kept[0][1].strip():
        kept.pop(0)
    while kept and not kept[-1][1].strip():
        kept.pop()
    return kept


def _to_documents(piece: _Piece, file_name: str, info: CodeInfo) -> List[Document]:
    documents = []
    for part, start in enumerate(range(0, len(piece.lines), MAX_CHUNK_LINES)):
        group = piece.lines[start:start + MAX_CHUNK_LINES]
        first, last = group[0][0], group[-1][0]
        line_numbers = {n for n, _ in group}
        header = f"# {file_name} | {piece.symbol} | L{first}-{last}"
        documents.append(Document(
            page_content=header + "\n" + "\n".join(text for _, text in group),
            metadata={
                "symbol": piece.symbol,
                "symbols": _members(piece, info.symbols, line_numbers),
                "start_line": first,
                "end_line": last,
                "language": info.language,
                "part": part,
            },
        ))
    return documents


def _members(piece: _Piece, symbols: List[Symbol], line_numbers) -> List[str]:
    """Symbols whose graph node belongs to this chunk: the chunk's own symbol (in every
    part, so each part links to it) plus every symbol that starts on one of its lines
    (methods, nested functions, class and instance variables)."""
    members = [piece.symbol]
    members += [s.qualname for s in symbols if s.qualname != piece.symbol and s.start_line in line_numbers]
    return members
