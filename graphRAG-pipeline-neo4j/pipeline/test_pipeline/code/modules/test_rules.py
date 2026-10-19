"""Tests of code_analysis/rules.py: the rule-based vulnerability and bug scanner. Offline."""
from pipeline.code_analysis.rules import scan_source
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON

VULNERABLE = '''import os
import pickle

PASSWORD = "hunter2hunter2"


def run(cmd, data):
    eval(cmd)
    os.system("ls " + cmd)
    return pickle.loads(data)
'''

CLEAN = '''import subprocess


def run(args):
    """Run a fixed command without a shell."""
    return subprocess.run(["ls", *args], check=True, capture_output=True)
'''


def test_flags_the_planted_problems():
    ids = {f.rule_id for f in scan_source(VULNERABLE, "app.py")}
    assert {"eval-exec", "shell-command", "hardcoded-secret"} <= ids
    assert any("deserial" in i for i in ids)


def test_clean_code_has_no_findings():
    assert scan_source(CLEAN, "app.py") == []


def test_finding_names_the_enclosing_function_and_line():
    info = extract(VULNERABLE, PYTHON)
    by_rule = {f.rule_id: f for f in scan_source(VULNERABLE, "app.py", info)}
    assert by_rule["eval-exec"].symbol == "run"
    assert by_rule["eval-exec"].start_line == 8
    assert by_rule["eval-exec"].detector == "rule"


def test_secret_value_is_masked_in_evidence():
    secret = next(f for f in scan_source(VULNERABLE, "app.py") if f.rule_id == "hardcoded-secret")
    assert "hunter2" not in secret.evidence and "***" in secret.evidence


def test_findings_are_ordered_by_line_and_ids_are_stable():
    first = scan_source(VULNERABLE, "app.py")
    assert [f.start_line for f in first] == sorted(f.start_line for f in first)
    assert [f.id for f in first] == [f.id for f in scan_source(VULNERABLE, "app.py")]
    assert all(f.id.startswith("F-") for f in first)


def test_syntax_error_becomes_a_finding_instead_of_raising():
    findings = scan_source("def broken(:\n    pass\n", "bad.py")
    assert len(findings) == 1 and findings[0].category == "syntax-error"
