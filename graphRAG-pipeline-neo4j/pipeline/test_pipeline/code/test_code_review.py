"""Tests for the LLM review (phase 2). A scripted fake LLM stands in for the model: no network, no cost."""
from pipeline.code_analysis.analyzer import analyze_file, risk_level
from pipeline.code_analysis.context import (
    SymbolContext, build_units, fetch_symbol_context, findings_in_unit, needs_review,
)
from pipeline.code_analysis.reviewer import (
    Dismissal, ReviewedFinding, ReviewOutput, build_review_prompt, merge_review, review_unit,
)
from pipeline.code_analysis.rules import scan_source
from pipeline.code_analysis.verify import clean_cwe, locate_evidence, redact
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.llm_setup.metrics import LLMMetrics

SOURCE = '''import os
import sqlite3

DB_PASSWORD = "hunter2hunter2"


def run_report(name):
    return os.system("report " + name)


def lookup(conn, user):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '" + user + "'")
    return cur.fetchall()


def add(a, b):
    return a + b
'''


class FakeLLM:
    """Same surface the reviewer uses (`astructured`, `metrics`), answers from a script."""

    def __init__(self, script):
        self.script = script          # {label prefix: output object or exception}
        self.labels = []
        self.prompts = {}
        self.metrics = LLMMetrics("gpt-4.1-mini")

    async def astructured(self, system_prompt, user_prompt, output_type, label=""):
        self.labels.append(label)
        self.prompts[label] = (system_prompt, user_prompt)
        self.metrics.record(label, 0.0, 0.1, None)
        for prefix, answer in self.script.items():
            if label.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer(user_prompt) if callable(answer) else answer
        return output_type() if label.startswith("review") else output_type(summary="Summary text.")


def _setup(source=SOURCE, file_name="app.py"):
    info = extract(source, PYTHON)
    return info, scan_source(source, file_name, info), build_units(source, info, file_name)


def _unit(units, symbol):
    return next(u for u in units if u.symbol == symbol)


def finding(**overrides):
    base = dict(kind="vulnerability", category="command-injection", cwe="CWE-78", severity="high",
                confidence="high", title="Shell command built from input", explanation="Name goes to the shell.",
                scenario="`x; rm -rf /` runs.", fix="Use subprocess with a list.", start_line=8, end_line=8,
                evidence='return os.system("report " + name)')
    return ReviewedFinding(**{**base, **overrides})


def test_units_are_functions_plus_module_level_code():
    _, _, units = _setup()
    assert [u.symbol for u in units] == ["run_report", "lookup", "add", None]
    assert _unit(units, "run_report").lines[0] == (7, "def run_report(name):")
    assert all(n in {7, 8} for n, _ in _unit(units, "run_report").lines)


def test_nested_functions_belong_to_their_outer_unit():
    source = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    _, _, units = _setup(source)
    assert [u.symbol for u in units] == ["outer"]
    assert len(units[0].lines) == 4


def test_only_flagged_or_risky_units_are_selected():
    _, findings, units = _setup()
    picked = {u.symbol for u in units if needs_review(u, findings_in_unit(u, findings), SymbolContext())}
    assert picked == {"run_report", "lookup", None}   # add() is trivial and untouched by rules; module has a secret


def test_entry_points_and_long_functions_are_selected():
    body = "\n".join(f"    x{i} = {i}" for i in range(6))
    source = f"def handler(request):\n{body}\n    return x0\n"
    _, findings, units = _setup(source)
    assert not needs_review(units[0], findings, SymbolContext())
    assert needs_review(units[0], findings, SymbolContext(entry_point=True))
    long_source = "def big():\n" + "\n".join(f"    y{i} = {i}" for i in range(45)) + "\n"
    _, findings, units = _setup(long_source)
    assert needs_review(units[0], findings, SymbolContext())


def test_context_is_read_from_the_graph():
    class Graph:
        def query(self, query, params):
            if "->(n:__Entity__)" in query:
                return [{"id": "db.py::run", "signature": "def run(sql)", "summary": "Runs SQL"}]
            return []

    context = fetch_symbol_context(Graph(), "app.py::lookup")
    assert context.callees == ["db.py::run: def run(sql) - Runs SQL"]
    assert context.callers == [] and context.entry_point is True   # nobody calls it
    assert fetch_symbol_context(Graph(), "app.py::lookup", decorated=True).entry_point is True


def test_prompt_has_numbered_code_context_and_candidates():
    _, findings, units = _setup()
    unit = _unit(units, "run_report")
    context = SymbolContext(callers=["main.py::main: def main()"], callees=[], entry_point=False)
    prompt = build_review_prompt(unit, findings_in_unit(unit, findings), context)
    assert "    8 |     return os.system(" in prompt
    assert prompt.index("<code>") < prompt.index("os.system") < prompt.index("</code>")
    assert "main.py::main" in prompt and "id F-" in prompt and "shell-command" not in prompt.split("<code>")[1]


def test_evidence_is_located_in_the_code():
    _, _, units = _setup()
    unit = _unit(units, "run_report")
    quote = 'return os.system("report " + name)'
    assert locate_evidence(unit, 8, 8, quote) == (8, 8)
    assert locate_evidence(unit, 8, 8, 'return   os.system("report "  + name)') == (8, 8)   # whitespace does not matter
    assert locate_evidence(unit, 7, 7, quote) == (8, 8)       # wrong line number, right quote: snapped to the real line
    assert locate_evidence(unit, 1, 1, "def run_report(name): return os.system") == (7, 8)   # a quote may span several lines
    assert locate_evidence(unit, 8, 8, "return eval(name)") is None                       # invented quote
    assert locate_evidence(unit, 8, 8, "   ") is None
    gutter_quote = '    7 | def run_report(name):' + chr(10) + '    8 |     return os.system("report " + name)'
    assert locate_evidence(unit, 7, 8, gutter_quote) == (7, 8)   # printed line numbers copied into the quote


def test_redaction_and_cwe_cleanup():
    assert "hunter2" not in redact('DB_PASSWORD = "hunter2hunter2" is exposed')
    assert redact('DB_PASSWORD = "hunter2hunter2"') == 'DB_PASSWORD = "***"'
    assert "AKIAABCDEFGHIJKLMNOP" not in redact("key AKIAABCDEFGHIJKLMNOP leaked")
    assert redact("nothing secret here") == "nothing secret here"
    assert clean_cwe("cwe-89") == "CWE-89" and clean_cwe("CWE-89 something") is None and clean_cwe(None) is None


def test_confirmed_rule_finding_gets_llm_explanation_and_keeps_its_lines():
    info, findings, units = _setup()
    unit = _unit(units, "run_report")
    rule = findings_in_unit(unit, findings)[0]
    output = ReviewOutput(findings=[finding(confirms=rule.id, severity="critical", explanation="Tailored text.",
                                            start_line=999, end_line=999, evidence="not checked for confirmations")])
    result = merge_review(unit, [rule], output, info)
    assert len(result.findings) == 1 and not result.dropped
    merged = result.findings[0]
    assert merged.id == rule.id and merged.detector == "rule+llm"
    assert merged.severity == "critical" and merged.explanation == "Tailored text."
    assert merged.start_line == rule.start_line and merged.evidence == rule.evidence


def test_dismissed_rule_finding_is_removed_and_reported():
    info, findings, units = _setup()
    unit = _unit(units, "run_report")
    rule = findings_in_unit(unit, findings)[0]
    result = merge_review(unit, [rule], ReviewOutput(dismissed=[Dismissal(id=rule.id, reason="constant input only")]), info)
    assert result.findings == []
    assert result.dismissed[0].finding.id == rule.id and result.dismissed[0].reason == "constant input only"


def test_unmentioned_rule_finding_stays():
    info, findings, units = _setup()
    unit = _unit(units, "run_report")
    rule = findings_in_unit(unit, findings)
    assert merge_review(unit, rule, ReviewOutput(), info).findings == rule


def test_new_llm_finding_is_verified_and_attributed():
    info, _, units = _setup()
    unit = _unit(units, "lookup")
    draft = finding(category="SQL Injection", cwe="cwe-89", start_line=13, end_line=13,
                    evidence="cur.execute(\"SELECT * FROM users WHERE name = '\" + user + \"'\")")
    result = merge_review(unit, [], ReviewOutput(findings=[draft]), info)
    new = result.findings[0]
    assert new.detector == "llm" and new.symbol == "lookup" and new.category == "sql-injection" and new.cwe == "CWE-89"
    assert new.id.startswith("F-") and new.file == "app.py"


def test_llm_restating_a_rule_hit_without_its_id_is_merged():
    info, findings, units = _setup()
    unit = _unit(units, "run_report")
    rule = findings_in_unit(unit, findings)[0]
    result = merge_review(unit, [rule], ReviewOutput(findings=[finding(cwe=rule.cwe, category="command-injection")]), info)
    assert [f.id for f in result.findings] == [rule.id] and result.findings[0].detector == "rule+llm"


def test_wrong_line_numbers_are_corrected_from_the_quote():
    info, _, units = _setup()
    unit = _unit(units, "lookup")
    off_by_one = finding(category="sql-injection", start_line=14, end_line=14, evidence="cur.execute(")
    result = merge_review(unit, [], ReviewOutput(findings=[off_by_one]), info)
    assert (result.findings[0].start_line, result.findings[0].end_line) == (13, 13)


def test_invented_findings_are_dropped():
    info, _, units = _setup()
    unit = _unit(units, "lookup")
    invented = finding(start_line=13, end_line=13, evidence="os.system(user)")
    result = merge_review(unit, [], ReviewOutput(findings=[invented]), info)
    assert result.findings == [] and len(result.dropped) == 1


def test_secret_values_in_llm_text_are_redacted():
    info, _, units = _setup()
    unit = _unit(units, None)
    draft = finding(kind="vulnerability", category="hardcoded-secret", start_line=4, end_line=4,
                    evidence='DB_PASSWORD = "hunter2hunter2"',
                    explanation='The line DB_PASSWORD = "hunter2hunter2" exposes the password.')
    new = merge_review(unit, [], ReviewOutput(findings=[draft]), info).findings[0]
    assert "hunter2" not in new.model_dump_json()


async def test_review_failure_keeps_rule_findings():
    info, findings, units = _setup()
    unit = _unit(units, "run_report")
    rule = findings_in_unit(unit, findings)
    llm = FakeLLM({"review": RuntimeError("rate limited")})
    result = await review_unit(llm, unit, rule, SymbolContext(), info)
    assert result.findings == rule and "rate limited" in result.error


async def test_analyze_file_end_to_end():
    info = extract(SOURCE, PYTHON)
    rule_ids = {f.symbol: f.id for f in scan_source(SOURCE, "app.py", info) if f.rule_id == "shell-command"}
    llm = FakeLLM({
        "review:run_report": ReviewOutput(findings=[finding(confirms=rule_ids["run_report"], severity="critical")]),
        "review:lookup": ReviewOutput(findings=[finding(
            category="sql-injection", cwe="CWE-89", start_line=13, end_line=13,
            evidence="cur.execute(\"SELECT * FROM users WHERE name = '\" + user + \"'\")")]),
        "summary": lambda prompt: __import__("pipeline.code_analysis.analyzer", fromlist=["x"])._SummaryText(
            summary="Command and SQL injection in run_report and lookup."),
    })
    report = await analyze_file(llm, SOURCE, "app.py", info)

    assert sorted(report.reviewed) == ["<module>", "add", "lookup", "run_report"]   # every function by default
    by_category = {f.category: f for f in report.findings}
    assert by_category["command-injection"].severity == "critical"
    assert by_category["sql-injection"].detector == "rule+llm"   # same line and category as the rule hit: merged, not duplicated
    assert len([f for f in report.findings if f.category == "sql-injection"]) == 1
    assert "hardcoded-secret" in by_category                     # the module's rule finding stays
    assert report.summary.risk_level == "critical"
    assert report.summary.text == "Command and SQL injection in run_report and lookup."
    assert "hunter2" not in "".join(p[1] for p in llm.prompts.values() if "Findings" in p[1])
    assert report.llm_metrics["calls"] == len(llm.labels) == 5   # 4 reviews + 1 summary


async def test_review_all_false_skips_trivial_functions():
    info = extract(SOURCE, PYTHON)
    llm = FakeLLM({})
    report = await analyze_file(llm, SOURCE, "app.py", info, review_all=False)
    assert sorted(report.reviewed) == ["<module>", "lookup", "run_report"]   # add() is trivial and untouched by rules
    assert not any(label.startswith("review:add") for label in llm.labels)


async def test_clean_file_reports_no_findings_and_no_summary_call():
    source = "def add(a, b):\n    return a + b\n"
    llm = FakeLLM({})
    report = await analyze_file(llm, source, "ok.py", extract(source, PYTHON))
    assert llm.labels == ["review:add"] and report.findings == []   # reviewed, nothing found: no summary call
    assert report.summary.risk_level == "none" and "not proof" in report.summary.text


async def test_syntax_error_skips_review_but_is_reported():
    source = "def f():\nreturn 1\n"
    llm = FakeLLM({})
    report = await analyze_file(llm, source, "bad.py", extract(source, PYTHON))
    assert [f.rule_id for f in report.findings] == ["syntax-error"]
    assert not any(label.startswith("review") for label in llm.labels)
    assert report.errors and report.summary.risk_level == "high"


async def test_summary_failure_falls_back_to_counts():
    source = "import os\nos.system(cmd)\n"
    llm = FakeLLM({"review": ReviewOutput(), "summary": RuntimeError("down")})
    report = await analyze_file(llm, source, "a.py", extract(source, PYTHON))
    assert report.summary.risk_level == "high" and "1 findings" in report.summary.text


def test_risk_level_is_the_highest_severity():
    source = "import os, hashlib\nhashlib.md5(x)\nos.system(cmd)\n"
    findings = scan_source(source, "a.py")
    assert risk_level(findings) == "high" and risk_level([]) == "none"
