"""Rule scan: cheap, exact checks on Python source, no LLM involved.

Every rule is a small generator `rule(node, ctx)` that yields a Hit for each problem it sees
at one AST node. `scan_source` walks the tree once, offers every node to every rule, and turns
hits into Findings using the RULE_INFO text. To add a rule: write the function, add it to
_RULES, and add its description to RULE_INFO.

Rules propose; they do not prove. The LLM reviewer (a later phase) confirms or rejects them
and finds what rules cannot, such as a logic error or a missing permission check."""
import ast
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from ..code_parsing.extractor import CodeInfo
from .models import Finding, make_finding_id

MAX_EVIDENCE_CHARS = 200


@dataclass(frozen=True)
class RuleInfo:
    kind: str
    category: str
    cwe: Optional[str]
    severity: str      # default; a rule can lower or raise it per hit
    confidence: str
    title: str
    explanation: str
    scenario: str
    fix: str


@dataclass
class Hit:
    rule_id: str
    node: ast.AST
    severity: Optional[str] = None   # overrides RuleInfo.severity
    evidence: Optional[str] = None   # overrides the source line (used to mask secrets)


@dataclass
class _Context:
    aliases: Dict[str, str]          # local name -> real dotted name ("sp" -> "subprocess")
    with_items: set                  # ids of expressions used as `with <expr>:`
    is_test: bool


RULE_INFO: Dict[str, RuleInfo] = {
    "syntax-error": RuleInfo(
        "bug", "syntax-error", None, "high", "high",
        "File does not compile",
        "Python cannot parse this file (bad syntax, indentation or mixed tabs and spaces).",
        "Importing or running the module raises the error immediately, so nothing in it works.",
        "Fix the syntax or indentation at the reported line."),
    "eval-exec": RuleInfo(
        "vulnerability", "code-injection", "CWE-95", "high", "high",
        "eval/exec on dynamic input",
        "eval() and exec() run a string as Python code. If any part of it comes from outside, the caller controls the program.",
        "An attacker sends input such as `__import__('os').system('...')` and runs commands on the server.",
        "Do not execute strings. Use ast.literal_eval for literals, or a lookup table or parser for the real need."),
    "shell-command": RuleInfo(
        "vulnerability", "command-injection", "CWE-78", "high", "high",
        "Shell command execution",
        "The command goes through a shell (os.system, os.popen or subprocess with shell=True), where characters like ; | & $() are interpreted.",
        "If user input reaches the command, `name; rm -rf /` runs a second command.",
        "Call subprocess.run([...]) with an argument list and shell=False, and validate any user-supplied values."),
    "unsafe-deserialization": RuleInfo(
        "vulnerability", "unsafe-deserialization", "CWE-502", "high", "high",
        "Unsafe deserialization",
        "pickle, marshal, dill, shelve and yaml.load without a safe loader can construct arbitrary objects while loading.",
        "A crafted payload runs code during loading, with the privileges of the process.",
        "Use json, or yaml.safe_load, and never unpickle data from an untrusted source."),
    "sql-injection": RuleInfo(
        "vulnerability", "sql-injection", "CWE-89", "high", "medium",
        "SQL built from a string",
        "The query is assembled with an f-string, % formatting, + or .format(), so values become part of the SQL text.",
        "Input like `' OR '1'='1` changes the query and can read, change or delete other rows.",
        "Pass values as parameters: cursor.execute('... WHERE id = %s', (value,))."),
    "hardcoded-secret": RuleInfo(
        "vulnerability", "hardcoded-secret", "CWE-798", "high", "medium",
        "Hardcoded credential",
        "A password, key or token is written into the source file as a string literal.",
        "Anyone with read access to the repository, including its history, has the credential.",
        "Move it to an environment variable or a secrets manager, and rotate the exposed value."),
    "weak-hash": RuleInfo(
        "vulnerability", "weak-hash", "CWE-327", "low", "medium",
        "Weak hash algorithm",
        "MD5 and SHA-1 are broken for security uses.",
        "If used for passwords or integrity checks, collisions or brute force defeat them.",
        "Use SHA-256 or better, and a password hash (bcrypt, argon2, scrypt) for passwords. Fine for non-security checksums."),
    "insecure-random": RuleInfo(
        "vulnerability", "insecure-random", "CWE-338", "medium", "medium",
        "Predictable random value used as a secret",
        "The random module is not cryptographically secure, and its output can be predicted.",
        "An attacker who observes some tokens can predict later ones and take over sessions or reset flows.",
        "Use the secrets module (secrets.token_urlsafe, secrets.token_hex)."),
    "tls-verify-disabled": RuleInfo(
        "vulnerability", "tls-verification-disabled", "CWE-295", "high", "high",
        "TLS certificate check disabled",
        "Certificate verification is turned off, so the client accepts any server certificate.",
        "A machine-in-the-middle presents its own certificate and reads or alters the traffic.",
        "Leave verification on, and pass a CA bundle for private certificates."),
    "world-writable": RuleInfo(
        "vulnerability", "insecure-permissions", "CWE-732", "medium", "high",
        "World-writable file permissions",
        "os.chmod gives write permission to every user on the machine.",
        "Another local user modifies the file, for example to plant code the program later runs.",
        "Use the narrowest mode needed, such as 0o600 or 0o644."),
    "assert-in-production": RuleInfo(
        "bug", "assert-for-validation", "CWE-617", "low", "medium",
        "assert used in non-test code",
        "assert statements are removed when Python runs with -O, so the check silently disappears.",
        "A validation that guards security or data integrity stops running in optimized deployments.",
        "Raise an explicit exception (ValueError, PermissionError) for checks that must always run."),
    "assert-tuple": RuleInfo(
        "bug", "assert-always-true", None, "medium", "high",
        "assert on a tuple is always true",
        "`assert (condition, 'message')` asserts a non-empty tuple, which is always truthy.",
        "The check never fails, so the bug it was meant to catch goes unnoticed.",
        "Write `assert condition, 'message'` without the parentheses."),
    "bare-except": RuleInfo(
        "bug", "bare-except", "CWE-396", "low", "high",
        "Bare except",
        "`except:` also catches KeyboardInterrupt and SystemExit, and hides the type of the failure.",
        "Ctrl+C or a shutdown request is swallowed, and real bugs are handled as if they were expected.",
        "Catch the specific exceptions you can handle, or `except Exception`."),
    "swallowed-exception": RuleInfo(
        "bug", "swallowed-exception", "CWE-390", "low", "medium",
        "Exception silently ignored",
        "The except block does nothing, so the error disappears without a log or re-raise.",
        "The program continues in a broken state and the cause is impossible to find later.",
        "Log the error, handle it, or re-raise it. If ignoring is intended, say why in a comment."),
    "mutable-default": RuleInfo(
        "bug", "mutable-default-argument", "CWE-1188", "medium", "high",
        "Mutable default argument",
        "A list, dict or set default is created once and shared by every call.",
        "Data added in one call shows up in the next call, causing hard-to-trace state leaks.",
        "Use `None` as the default and create the object inside the function."),
    "open-without-with": RuleInfo(
        "bug", "resource-leak", "CWE-772", "low", "medium",
        "File opened without `with`",
        "The file is not closed automatically if an error occurs before close() is reached.",
        "Under load or repeated errors the process runs out of file handles.",
        "Use `with open(...) as f:`."),
    "is-literal": RuleInfo(
        "bug", "identity-comparison-with-literal", None, "medium", "high",
        "`is` compared with a literal",
        "`is` tests object identity, not equality. Whether two equal literals are the same object is an implementation detail.",
        "The comparison is true in some runs and false in others.",
        "Use == or != for strings, numbers and bytes. Keep `is` for None, True and False."),
    "unreachable-code": RuleInfo(
        "bug", "unreachable-code", "CWE-561", "low", "high",
        "Code after return, raise, break or continue",
        "These statements follow one that always leaves the block, so they never run.",
        "Often a sign of an unfinished edit or a wrongly indented line, and the intended logic does not happen.",
        "Remove the code, or fix the indentation or ordering."),
}


SECRET_NAME = re.compile(r"password|passwd|pwd|secret|token|api_?key|private_?key|access_?key", re.I)
NOT_A_SECRET_NAME = re.compile(r"(_url|_uri|_type|_field|_name|_path|_file|_header|_prefix|_endpoint)$", re.I)
KNOWN_CREDENTIAL = re.compile(r"AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
PLACEHOLDER = re.compile(r"^(<.*>|\$\{.*\}|\{\{.*\}\}|x+|\*+|changeme|your[_-].*|example.*|todo|none|null|test|dummy)$", re.I)
SECRET_PURPOSE = re.compile(r"token|secret|password|passwd|otp|nonce|session|key", re.I)

SUBPROCESS_CALLS = {"subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output", "subprocess.Popen"}
UNSAFE_LOADERS = {"pickle.load", "pickle.loads", "cPickle.load", "cPickle.loads", "marshal.load", "marshal.loads",
                  "dill.load", "dill.loads", "shelve.open"}
SQL_METHODS = {"execute", "executemany", "executescript", "raw"}


def _dotted_name(node, ctx: _Context) -> str:
    """"sp.run" -> "subprocess.run" when `import subprocess as sp`; "" if not a plain name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(ctx.aliases.get(node.id, node.id))
    return ".".join(reversed(parts))


def _is_str_constant(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _first_arg_is_dynamic(call: ast.Call) -> bool:
    return bool(call.args) and not isinstance(call.args[0], ast.Constant)


def _keyword(call: ast.Call, name: str):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _builds_string(node) -> bool:
    """f-string, "..." % x, "..." + x or "...".format(x): text assembled from a value."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        sides = (node.left, node.right)
        return any(_is_str_constant(s) or isinstance(s, ast.JoinedStr) for s in sides) and not all(
            isinstance(s, ast.Constant) for s in sides
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _is_str_constant(node.func.value) and bool(node.args or node.keywords)
    return False


def _named_values(node):
    """(name, value) pairs a node assigns: `x = v`, `x: T = v`, `f(x=v)`, `{"x": v}`."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            if name and node.value is not None:
                yield name, node.value
    elif isinstance(node, ast.keyword) and node.arg:
        yield node.arg, node.value
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if _is_str_constant(key):
                yield key.value, value


def _is_secret_literal(value) -> bool:
    return _is_str_constant(value) and len(value.value.strip()) >= 6 and not PLACEHOLDER.match(value.value.strip())


def _eval_exec(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call) and _dotted_name(node.func, ctx) in ("eval", "exec"):
        yield Hit("eval-exec", node, severity="high" if _first_arg_is_dynamic(node) else "low")


def _shell_command(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, ast.Call):
        return
    name = _dotted_name(node.func, ctx)
    shell = _keyword(node, "shell")
    uses_shell = isinstance(shell, ast.Constant) and shell.value is True
    if name in ("os.system", "os.popen") or (name in SUBPROCESS_CALLS and uses_shell):
        yield Hit("shell-command", node, severity="high" if _first_arg_is_dynamic(node) else "medium")


def _unsafe_deserialization(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, ast.Call):
        return
    name = _dotted_name(node.func, ctx)
    if name in UNSAFE_LOADERS:
        yield Hit("unsafe-deserialization", node)
    elif name in ("yaml.load", "yaml.load_all"):
        loader = _keyword(node, "Loader") or (node.args[1] if len(node.args) > 1 else None)
        if loader is None or not _dotted_name(loader, ctx).endswith("SafeLoader"):
            yield Hit("unsafe-deserialization", node)


def _sql_injection(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SQL_METHODS:
        if node.args and _builds_string(node.args[0]):
            yield Hit("sql-injection", node)


def _hardcoded_secret(node, ctx) -> Iterator[Hit]:
    for name, value in _named_values(node):
        if SECRET_NAME.search(name) and not NOT_A_SECRET_NAME.search(name) and _is_secret_literal(value):
            yield Hit("hardcoded-secret", node, evidence=f'{name} = "***"')  # the value is never copied
            return
    if _is_str_constant(node) and KNOWN_CREDENTIAL.search(node.value):
        yield Hit("hardcoded-secret", node, evidence='known credential format "***"')


def _weak_hash(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call) and _dotted_name(node.func, ctx) in ("hashlib.md5", "hashlib.sha1"):
        yield Hit("weak-hash", node)


def _insecure_random(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return
    for name, value in _named_values(node):
        if SECRET_PURPOSE.search(name) and any(isinstance(n, ast.Call) and _dotted_name(n.func, ctx).startswith("random.") for n in ast.walk(value)):
            yield Hit("insecure-random", node)
            return


def _tls_verify_disabled(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call):
        verify = _keyword(node, "verify")
        if isinstance(verify, ast.Constant) and verify.value is False:
            yield Hit("tls-verify-disabled", node)
        elif _dotted_name(node.func, ctx) == "ssl._create_unverified_context":
            yield Hit("tls-verify-disabled", node)
    elif isinstance(node, ast.Attribute) and _dotted_name(node, ctx) == "ssl.CERT_NONE":
        yield Hit("tls-verify-disabled", node)


def _world_writable(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call) and _dotted_name(node.func, ctx) == "os.chmod" and len(node.args) > 1:
        mode = node.args[1]
        if isinstance(mode, ast.Constant) and isinstance(mode.value, int) and mode.value & 0o002:
            yield Hit("world-writable", node)


def _assert_rules(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, ast.Assert):
        return
    if isinstance(node.test, ast.Tuple) and node.test.elts:
        yield Hit("assert-tuple", node)
    elif not ctx.is_test:
        yield Hit("assert-in-production", node)


def _except_rules(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, ast.ExceptHandler):
        return
    if node.type is None:
        yield Hit("bare-except", node)
    only_pass = all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in node.body)
    if only_pass:
        yield Hit("swallowed-exception", node)


def _mutable_default(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
        is_literal = isinstance(default, (ast.List, ast.Dict, ast.Set))
        is_constructor = isinstance(default, ast.Call) and _dotted_name(default.func, ctx) in ("list", "dict", "set")
        if is_literal or is_constructor:
            yield Hit("mutable-default", default)
            return


def _open_without_with(node, ctx) -> Iterator[Hit]:
    if isinstance(node, ast.Call) and _dotted_name(node.func, ctx) in ("open", "io.open") and id(node) not in ctx.with_items:
        yield Hit("open-without-with", node)


def _is_literal(node, ctx) -> Iterator[Hit]:
    if not isinstance(node, ast.Compare):
        return
    operands = [node.left] + node.comparators
    for op, left, right in zip(node.ops, operands, operands[1:]):
        if isinstance(op, (ast.Is, ast.IsNot)) and any(_is_plain_literal(side) for side in (left, right)):
            yield Hit("is-literal", node)
            return


def _is_plain_literal(node) -> bool:
    """str, bytes and number literals. None, True, False and ... are meant to be compared with `is`."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes, int, float)) and not isinstance(node.value, bool)


def _unreachable_code(node, ctx) -> Iterator[Hit]:
    for field_name in ("body", "orelse", "finalbody"):
        statements = getattr(node, field_name, None)
        if not isinstance(statements, list):
            continue
        for index, statement in enumerate(statements[:-1]):
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                yield Hit("unreachable-code", statements[index + 1])
                break


_RULES = [
    _eval_exec, _shell_command, _unsafe_deserialization, _sql_injection, _hardcoded_secret, _weak_hash,
    _insecure_random, _tls_verify_disabled, _world_writable, _assert_rules, _except_rules, _mutable_default,
    _open_without_with, _is_literal, _unreachable_code,
]


def scan_source(source: str, file_name: str, info: Optional[CodeInfo] = None) -> List[Finding]:
    """Run every rule over Python source. `info` (from the extractor) lets each finding
    name the function or class it is in. Returns findings ordered by line."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:  # IndentationError and TabError are subclasses
        return [_syntax_finding(error, source, file_name, info)]

    ctx = _Context(
        aliases=_import_aliases(tree),
        with_items=_with_items(tree),
        is_test=_is_test_file(file_name),
    )
    lines = source.splitlines()
    findings, seen = [], set()
    for node in ast.walk(tree):
        for rule in _RULES:
            for hit in rule(node, ctx):
                start = getattr(hit.node, "lineno", 1)
                if (hit.rule_id, start) in seen:  # e.g. a secret found by its name and again by its format
                    continue
                seen.add((hit.rule_id, start))
                findings.append(_to_finding(hit, start, lines, file_name, info))
    return sorted(findings, key=lambda f: (f.start_line, f.rule_id))


def _to_finding(hit: Hit, start: int, lines: List[str], file_name: str, info: Optional[CodeInfo]) -> Finding:
    rule = RULE_INFO[hit.rule_id]
    symbol = symbol_at(info, start)
    evidence = hit.evidence or (lines[start - 1].strip() if start <= len(lines) else "")
    return Finding(
        id=make_finding_id(file_name, symbol, hit.rule_id, start),
        kind=rule.kind, category=rule.category, cwe=rule.cwe,
        severity=hit.severity or rule.severity, confidence=rule.confidence,
        title=rule.title, explanation=rule.explanation, scenario=rule.scenario, fix=rule.fix,
        file=file_name, symbol=symbol,
        start_line=start, end_line=getattr(hit.node, "end_lineno", start) or start,
        evidence=evidence[:MAX_EVIDENCE_CHARS],
        detector="rule", rule_id=hit.rule_id,
    )


def _syntax_finding(error: SyntaxError, source: str, file_name: str, info: Optional[CodeInfo]) -> Finding:
    rule = RULE_INFO["syntax-error"]
    line = error.lineno or 1
    lines = source.splitlines()
    symbol = symbol_at(info, line)
    return Finding(
        id=make_finding_id(file_name, symbol, "syntax-error", line),
        kind=rule.kind, category=rule.category, cwe=rule.cwe,
        severity=rule.severity, confidence=rule.confidence,
        title=f"{type(error).__name__}: {error.msg}",
        explanation=f"{rule.explanation} Python reports: {error.msg} (line {line}). "
                    "Symbols extracted from this file may be incomplete or wrong.",
        scenario=rule.scenario, fix=rule.fix,
        file=file_name, symbol=symbol, start_line=line, end_line=line,
        evidence=(lines[line - 1].strip() if line <= len(lines) else "")[:MAX_EVIDENCE_CHARS],
        detector="rule", rule_id="syntax-error",
    )


def symbol_at(info: Optional[CodeInfo], line: int) -> Optional[str]:
    """Innermost function or class containing the line, None for module-level code."""
    if info is None:
        return None
    inside = [s for s in info.symbols if s.kind != "Variable" and s.start_line <= line <= s.end_line]
    return min(inside, key=lambda s: s.end_line - s.start_line).qualname if inside else None


def _import_aliases(tree: ast.AST) -> Dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _with_items(tree: ast.AST) -> set:
    return {
        id(item.context_expr)
        for node in ast.walk(tree) if isinstance(node, (ast.With, ast.AsyncWith))
        for item in node.items
    }


def _is_test_file(file_name: str) -> bool:
    name = file_name.replace("\\", "/").split("/")[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
