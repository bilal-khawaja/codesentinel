"""Tests of code_analysis helpers: finding ids, review units, evidence checks, redaction. Offline."""
from pipeline.code_analysis.context import build_units
from pipeline.code_analysis.models import make_finding_id
from pipeline.code_analysis.verify import clean_cwe, locate_evidence, normalize, redact
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON

SOURCE = '''def charge(amount):
    total = amount * 2
    return total


def refund(amount):
    return -amount
'''


def test_finding_id_is_stable_and_depends_on_location():
    a = make_finding_id("a.py", "f", "eval-exec", 3)
    assert a == make_finding_id("a.py", "f", "eval-exec", 3)
    assert a.startswith("F-") and len(a) == 12
    assert a != make_finding_id("a.py", "f", "eval-exec", 4)
    assert a != make_finding_id("b.py", "f", "eval-exec", 3)


def test_clean_cwe_accepts_only_the_cwe_format():
    assert clean_cwe(" cwe-89 ") == "CWE-89"
    assert clean_cwe("sql injection") is None
    assert clean_cwe(None) is None


def test_redact_removes_secret_values():
    assert "hunter2hunter2" not in redact('password = "hunter2hunter2"')
    assert redact("nothing secret here") == "nothing secret here"


def test_normalize_collapses_whitespace():
    assert normalize("a   b\n  c") == "a b c"


def test_each_function_becomes_a_review_unit():
    units = build_units(SOURCE, extract(SOURCE, PYTHON), "pay.py")
    by_symbol = {u.symbol: u for u in units}
    assert {"charge", "refund"} <= set(by_symbol)
    assert (by_symbol["charge"].start_line, by_symbol["charge"].end_line) == (1, 3)


def test_evidence_is_located_even_when_cited_lines_are_off():
    units = build_units(SOURCE, extract(SOURCE, PYTHON), "pay.py")
    charge = next(u for u in units if u.symbol == "charge")
    assert locate_evidence(charge, 1, 1, "total = amount * 2") == (2, 2)  # quote is trusted over the cited line
    assert locate_evidence(charge, 1, 1, "os.system(user_input)") is None  # invented evidence
