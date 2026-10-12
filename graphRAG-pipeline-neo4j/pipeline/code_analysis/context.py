"""What the reviewer is shown: the code of one unit plus where it sits in the call graph."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..code_parsing.chunker import lines_outside
from ..code_parsing.extractor import CodeInfo, Symbol
from .models import Finding

MAX_REVIEW_LINES = 250        # longer units are cut, and the review says so
LONG_FUNCTION_LINES = 40      # functions this long are reviewed even if nothing flagged them
MIN_ENTRY_POINT_LINES = 5
CONTEXT_LIMIT = 10            # callers and callees listed per unit

# Calls that touch the outside world: a function using one is worth a review.
SINK_HEADS = {"os", "subprocess", "pickle", "yaml", "marshal", "requests", "httpx", "urllib", "socket",
              "shutil", "sqlite3", "tempfile", "ftplib", "smtplib", "eval", "exec", "open"}
SINK_METHODS = {"execute", "executemany", "executescript", "raw", "system", "popen", "urlopen", "request",
                "sendall", "unlink", "rmtree", "chmod", "render_template_string"}


@dataclass
class SymbolContext:
    """Graph neighbours of a unit, as short readable lines ("file::name: signature - summary")."""
    callers: List[str] = field(default_factory=list)
    callees: List[str] = field(default_factory=list)
    entry_point: bool = False
    known: bool = False   # True when read from the graph; False = the file was analyzed on its own


@dataclass
class ReviewUnit:
    file: str
    symbol: Optional[str]                 # qualname, None = the module-level code of the file
    signature: str
    summary: str
    lines: List[Tuple[int, str]]          # (line number, text), what the reviewer sees
    truncated: bool = False
    calls: List[str] = field(default_factory=list)
    decorated: bool = False
    imports: List[str] = field(default_factory=list)   # names the file imports ("time", "Path", ...)
    defined: List[str] = field(default_factory=list)   # top-level functions, classes and variables of the file

    @property
    def line_numbers(self) -> set:
        return {n for n, _ in self.lines}

    @property
    def start_line(self) -> int:
        return self.lines[0][0]

    @property
    def end_line(self) -> int:
        return self.lines[-1][0]


def build_units(source: str, info: CodeInfo, file_name: str, summaries: Optional[Dict[str, str]] = None) -> List[ReviewUnit]:
    """One unit per function or method (nested functions are part of their outer function)
    plus one for the module-level code. Classes themselves are covered through their methods."""
    summaries = summaries or {}
    all_lines = source.splitlines()
    by_qualname = {s.qualname: s for s in info.symbols}
    imports = imported_names(info)
    defined = sorted({s.name for s in info.symbols if s.parent is None})
    units = []
    for symbol in info.symbols:
        if symbol.kind != "Function" or _inside_function(symbol, by_qualname):
            continue
        numbered = [(n, all_lines[n - 1]) for n in range(symbol.start_line, symbol.end_line + 1)]
        units.append(ReviewUnit(
            file=file_name, symbol=symbol.qualname, signature=symbol.signature,
            summary=symbol.docstring or summaries.get(symbol.qualname, ""),
            lines=numbered[:MAX_REVIEW_LINES], truncated=len(numbered) > MAX_REVIEW_LINES,
            calls=symbol.calls, decorated=all_lines[symbol.start_line - 1].lstrip().startswith("@"),
            imports=imports, defined=defined,
        ))

    top_level = [(s.start_line, s.end_line) for s in info.symbols if s.parent is None and s.kind != "Variable"]
    module_lines = lines_outside([(n, text) for n, text in enumerate(all_lines, 1)], top_level)
    if module_lines:
        units.append(ReviewUnit(
            file=file_name, symbol=None, signature="module-level code of " + file_name,
            summary=info.module_docstring, lines=module_lines[:MAX_REVIEW_LINES],
            truncated=len(module_lines) > MAX_REVIEW_LINES, imports=imports, defined=defined,
        ))
    return units


def imported_names(info: CodeInfo) -> List[str]:
    """Every name the file's imports make available: alias, module (and its first part), imported name."""
    names = set()
    for entry in info.imports:
        alias, module, name = (entry.split("|") + ["", "", ""])[:3]
        for n in (alias, module, module.split(".")[0], name):
            if n and n != "*":
                names.add(n)
    return sorted(names)


def _inside_function(symbol: Symbol, by_qualname: Dict[str, Symbol]) -> bool:
    parent = by_qualname.get(symbol.parent) if symbol.parent else None
    while parent is not None:
        if parent.kind == "Function":
            return True
        parent = by_qualname.get(parent.parent) if parent.parent else None
    return False


def findings_in_unit(unit: ReviewUnit, findings: List[Finding]) -> List[Finding]:
    return [f for f in findings if f.start_line in unit.line_numbers]


def needs_review(unit: ReviewUnit, rule_findings: List[Finding], context: SymbolContext) -> bool:
    """Spend an LLM call only where it can pay off: the scanner flagged something, the code
    touches files/network/shell/database, it is reachable from outside, or it is long."""
    if rule_findings:
        return True
    if unit.symbol is None:
        return False  # module-level code is only reviewed when the scanner flagged it
    if _calls_a_sink(unit.calls):
        return True
    size = len(unit.lines)
    if size >= LONG_FUNCTION_LINES:
        return True
    return size >= MIN_ENTRY_POINT_LINES and (context.entry_point or unit.decorated)


def _calls_a_sink(calls: List[str]) -> bool:
    for call in calls:
        parts = call.split(".")
        if parts[0] in SINK_HEADS or (len(parts) > 1 and parts[-1] in SINK_METHODS):
            return True
    return False


CALLEES_QUERY = """
MATCH (:__Entity__ {id: $id})-[r:RELATED {type: 'CALLS'}]->(n:__Entity__)
RETURN n.id AS id, n.signature AS signature, n.summary AS summary LIMIT $limit
"""

CALLERS_QUERY = """
MATCH (n:__Entity__)-[r:RELATED {type: 'CALLS'}]->(:__Entity__ {id: $id})
RETURN n.id AS id, n.signature AS signature, n.summary AS summary LIMIT $limit
"""


def fetch_symbol_context(graph, entity_id: str, decorated: bool = False) -> SymbolContext:
    """Callers and callees of one function entity, read from the graph (needs the link
    step to have run). A function nobody calls, or a decorated one (routes, CLI commands),
    counts as an entry point."""
    params = {"id": entity_id, "limit": CONTEXT_LIMIT}
    callees = [_describe(row) for row in graph.query(CALLEES_QUERY, params)]
    callers = [_describe(row) for row in graph.query(CALLERS_QUERY, params)]
    return SymbolContext(callers=callers, callees=callees, entry_point=decorated or not callers, known=True)


def _describe(row: dict) -> str:
    detail = " - ".join(part for part in (row.get("signature"), row.get("summary")) if part)
    return f"{row['id']}: {detail}" if detail else row["id"]
