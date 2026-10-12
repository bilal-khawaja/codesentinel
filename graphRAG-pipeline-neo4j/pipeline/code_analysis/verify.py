"""Checks on what the LLM says, done in plain code: models invent line numbers and quotes."""
import ast
import re
import textwrap
from typing import Optional, Tuple

from .context import ReviewUnit
from .rules import KNOWN_CREDENTIAL

CWE_FORMAT = re.compile(r"^CWE-\d{1,4}$")
# NAME = "value" / NAME: "value" where NAME looks like a secret: hide the value
SECRET_ASSIGNMENT = re.compile(r"""((?:password|passwd|pwd|secret|token|api_?key|private_?key|access_?key)\w*['"]?\s*[:=]\s*)(['"])[^'"\n]{4,}\2""", re.I)


LINE_NUMBER_GUTTER = re.compile(r"^\s*\d+\s*\|\s?", re.M)   # the "  17 | " the prompt prints before each line


def normalize(text: str) -> str:
    return " ".join(text.split())


MAX_QUOTE_LINES = 6


def locate_evidence(unit: ReviewUnit, start_line: int, end_line: int, evidence: str) -> Optional[Tuple[int, int]]:
    """Where the quoted evidence really is in the unit, as (start, end) line numbers.
    Models often get a line number off by one but quote the code correctly, so the quote
    is trusted over the numbers: first at the cited lines, then anywhere in the unit.
    None means the quote is nowhere in the code, i.e. the finding was invented."""
    quote = normalize(LINE_NUMBER_GUTTER.sub("", evidence))  # models often copy the gutter into the quote
    if not quote:
        return None
    cited = [text for n, text in unit.lines if start_line <= n <= end_line]
    if cited and quote in normalize(" ".join(cited)):
        return start_line, end_line
    for size in range(1, MAX_QUOTE_LINES + 1):
        for index in range(len(unit.lines) - size + 1):
            window = unit.lines[index:index + size]
            if quote in normalize(" ".join(text for _, text in window)):
                return window[0][0], window[-1][0]
    return None


def redact(text: str) -> str:
    """Remove secret values from text the LLM wrote, so the graph never stores one."""
    text = KNOWN_CREDENTIAL.sub("***", text)
    return SECRET_ASSIGNMENT.sub(lambda m: f'{m.group(1)}{m.group(2)}***{m.group(2)}', text)


def clean_cwe(cwe: Optional[str]) -> Optional[str]:
    return cwe.strip().upper() if cwe and CWE_FORMAT.match(cwe.strip().upper()) else None



QUERY_INJECTION = re.compile(r"sql|cypher|cql|nosql|query", re.I)
COMMAND_INJECTION = re.compile(r"command|shell", re.I)
DESERIALIZATION = re.compile(r"deserial", re.I)
UNSAFE_LOADERS = {"pickle", "cpickle", "_pickle", "dill", "cloudpickle", "jsonpickle", "marshal", "shelve", "yaml", "joblib", "torch"}
SHELL_CALLS = {"os.system", "os.popen", "os.spawnl", "os.spawnlp", "os.spawnv", "commands.getoutput", "pty.spawn"}
MISSING_NAME = re.compile(r"import|undefined|undeclared|name-?error|not-defined", re.I)
PRAISE = re.compile(r"\b(proper(ly)?|safe(ly)?|correct(ly)?|secure(ly)?|good|well[- ]handled)\b", re.I)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def contradiction(draft, unit: ReviewUnit) -> Optional[str]:
    """A new LLM finding the file itself disproves, checked in plain code. Returns the reason
    to drop it, or None. Two known model mistakes: calling a fixed query string an injection,
    and reporting an import or name that the file does have."""
    if PRAISE.search(draft.title) or draft.severity == "info":
        return "not a problem: the model described something that is fine"
    category = draft.category.lower()
    if QUERY_INJECTION.search(category) and "injection" in category and not builds_strings(unit):
        return "no string is built or formatted in this code, so the query text is fixed (parameters are not injection)"
    if COMMAND_INJECTION.search(category) and not uses_shell(unit):
        return "no shell is involved (no os.system/popen, no shell=True), arguments are passed as a list"
    if DESERIALIZATION.search(category) and not uses_unsafe_deserializer(unit):
        return "only a safe data format (json) is parsed here, no pickle/yaml/marshal-style loader"
    if MISSING_NAME.search(category):
        known = set(unit.imports) | set(unit.defined)
        claimed = known & set(_WORD.findall(draft.title))
        if claimed:
            return f"the file does import or define {', '.join(sorted(claimed))}"
    return None


LOG_HEADS = {"logging", "logger", "log", "print"}


def builds_strings(unit: ReviewUnit) -> bool:
    """True if the code assembles text dynamically (f-string with values, + or % on strings,
    .format, +=), ignoring text that only goes to a log or print call. Unparseable code
    counts as True, so nothing is dropped on a guess."""
    try:
        tree = ast.parse(textwrap.dedent(chr(10).join(text for _, text in unit.lines)))
    except SyntaxError:
        return True
    ignored = set()  # nodes inside logging/print calls: building a log message is not building a query
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_head(node.func) in LOG_HEADS:
            ignored |= {id(n) for arg in node.args + [k.value for k in node.keywords] for n in ast.walk(arg)}
    for node in ast.walk(tree):
        if id(node) in ignored:
            continue
        if isinstance(node, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) for v in node.values):
            return True
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, (ast.Add, ast.Mod)):
            operands = [node.left, node.right] if isinstance(node, ast.BinOp) else [node.value]
            if any(isinstance(o, ast.JoinedStr) or (isinstance(o, ast.Constant) and isinstance(o.value, str)) for o in operands):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "format_map"):
            return True
    return False


def _call_head(func) -> str:
    """First name of a dotted call: logging.error -> logging, self.logger.info -> self."""
    while isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            return func.attr if func.attr in LOG_HEADS else "self"
        func = func.value
    return func.id if isinstance(func, ast.Name) else ""


def _parsed_calls(unit: ReviewUnit):
    """(dotted name, ast.Call) for every call in the unit, or None when the code does not parse."""
    try:
        tree = ast.parse(textwrap.dedent(chr(10).join(text for _, text in unit.lines)))
    except SyntaxError:
        return None
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts, func = [], node.func
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            calls.append((".".join(reversed(parts)), node))
    return calls


def uses_shell(unit: ReviewUnit) -> bool:
    """True if the code can hand a string to a shell. Unparseable code, eval/exec and code that
    imports a wrapper we cannot see through count as True, so nothing is dropped on a guess."""
    calls = _parsed_calls(unit)
    if calls is None:
        return True
    for name, node in calls:
        if name in SHELL_CALLS or name in ("eval", "exec", "getoutput", "popen"):
            return True
        if any(k.arg == "shell" and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in node.keywords):
            return True
        if name.startswith("subprocess.") and node.args and isinstance(node.args[0], (ast.Name, ast.JoinedStr, ast.BinOp)):
            return True  # a non-list command (string or variable) may be split or shell-run: keep the finding
    return False


LOADER_METHODS = {"load", "loads", "load_all", "unsafe_load", "full_load", "loader"}


def uses_unsafe_deserializer(unit: ReviewUnit) -> bool:
    """True if the unit calls a loader that can run code from data (pickle, yaml, marshal ...), or
    a load/loads call on a name we cannot rule out (an alias such as `pk.loads`). json is safe."""
    calls = _parsed_calls(unit)
    if calls is None:
        return True
    for name, _ in calls:
        head, _, tail = name.partition(".")
        tail = name.rsplit(".", 1)[-1]
        if head.lower() in UNSAFE_LOADERS:
            return True
        if tail in LOADER_METHODS and head.lower() not in ("json", "self", "cls", "tomllib", "csv"):
            return True
    return False
