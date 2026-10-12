"""Tests of the code review: the real analyzer with the real LLM on the files in samples/.

    samples/vulnerable_app.py   10 planted vulnerabilities and bugs (labels: VULNERABLE_EXPECTED)
    samples/clean_app.py        the safe version of the same code, nothing planted

No fake model and no inline source: every test reads its code from samples/. The LLM's time, tokens and
cost are printed for each file (run with -s, which `just test-review` does). About $0.03 per session;
skipped when no model is configured. The model is not deterministic, so the thresholds sit just outside
what repeated live runs measured: they catch regressions, not one unlucky answer."""
import asyncio
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pipeline.code_analysis.analyzer import analyze_file
from pipeline.code_analysis.evaluation import VULNERABLE_EXPECTED, Sample, score_findings
from pipeline.code_analysis.rules import scan_source
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.graphdb.db import get_value_from_env
from pipeline.llm_setup.llm import get_llm

SAMPLES = Path(__file__).resolve().parent / "samples"
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
MODEL = "openai_gpt_4_1_mini"

MIN_RECALL = 0.8          # of the 10 planted problems; the rule scan alone finds 7
MIN_PRECISION = 0.5       # of everything reported on both files; measured 53-83%
MAX_CLEAN_FINDINGS = 8    # on the clean file; measured 2-7 doubtful findings (SSRF, None checks)
PLANTED_SECRET = ""


@pytest.fixture(scope="module")
def new_llm():
    """A fresh model handle per analysis. Each asyncio.run() gets its own event loop, and a handle's HTTP
    client is bound to the loop it first ran on, so handles (and their metrics) are not shared."""
    if not get_value_from_env(f"LLM_MODEL_CONFIG_{MODEL.upper()}"):
        pytest.skip(f"no LLM configured for {MODEL}")
    return lambda: get_llm(MODEL)[0]


def analyze_sample(new_llm, name, expected, review_all=True):
    """Analyze samples/<name> with the real model. Returns the sample, the report and the model handle."""
    source = (SAMPLES / name).read_text(encoding="utf-8")
    llm = new_llm()
    report = asyncio.run(analyze_file(llm, source, name, extract(source, PYTHON), review_all=review_all))
    print(f"\n--- LLM metrics: {name} (review_all={review_all}) ---\n{llm.metrics.report()}")
    return Sample(name, source, expected), report, llm


@pytest.fixture(scope="module")
def vulnerable(new_llm):
    return analyze_sample(new_llm, "vulnerable_app.py", VULNERABLE_EXPECTED)


@pytest.fixture(scope="module")
def clean(new_llm):
    return analyze_sample(new_llm, "clean_app.py", [])


def test_finds_the_planted_problems(vulnerable):
    sample, report, _ = vulnerable
    score = score_findings(sample, report.findings)
    print(f"\nvulnerable_app.py: TP {score.true_positives} FN {score.false_negatives} FP {score.false_positives} "
          f"missed={score.missed} extra={score.extra}")
    assert not report.errors, report.errors
    assert score.recall >= MIN_RECALL, f"missed: {score.missed}"


def test_llm_adds_what_the_rules_cannot_see(vulnerable):
    """Index errors, None dereferences and path traversal need understanding, not a pattern."""
    sample, report, _ = vulnerable
    rules = score_findings(sample, scan_source(sample.source, sample.name, extract(sample.source, PYTHON)))
    assert score_findings(sample, report.findings).recall > rules.recall, f"rules missed {rules.missed}"
    assert any(f.detector in ("llm", "rule+llm") for f in report.findings)


def test_every_function_is_reviewed_and_findings_are_well_formed(vulnerable):
    sample, report, _ = vulnerable
    functions = {s.qualname for s in extract(sample.source, PYTHON).symbols if s.kind == "Function"}
    assert functions <= set(report.reviewed)
    line_count = len(sample.source.splitlines())
    for f in report.findings:
        assert 1 <= f.start_line <= f.end_line <= line_count
        assert f.evidence and f.explanation and f.fix and f.id.startswith("F-")
    assert report.summary.risk_level in ("critical", "high")   # computed from the findings, never taken from the model
    assert report.summary.text


def test_secret_value_never_reaches_findings_or_summary(vulnerable):
    _, report, _ = vulnerable
    stored = " ".join(f.model_dump_json() for f in report.findings) + report.summary.text
    assert PLANTED_SECRET not in stored and "51HqT7bK2m9X" not in stored


def test_metrics_are_recorded_for_every_llm_call(vulnerable, clean):
    for _, report, llm in (vulnerable, clean):
        summary = llm.metrics.summary()
        assert report.llm_metrics == summary
        assert summary["calls"] >= len(report.reviewed)           # one per unit, plus the file summary
        assert summary["input_tokens"] > 0 and summary["output_tokens"] > 0
        assert summary["total_tokens"] == summary["input_tokens"] + summary["output_tokens"]
        assert 0 < summary["cost_usd"] < 0.25 and summary["cost_complete"]
        assert summary["total_seconds"] > 0 and summary["wall_seconds"] <= summary["total_seconds"] + 0.01
        assert llm.metrics.report()


def test_clean_file_is_not_flooded_with_findings(clean):
    _, report, _ = clean
    print(f"\nclean_app.py: {len(report.findings)} findings: {[f.title for f in report.findings]}")
    assert not report.errors, report.errors
    assert len(report.findings) <= MAX_CLEAN_FINDINGS


def test_clean_file_has_no_secret_or_sql_injection_findings(clean):
    """The classes that were false positives before the prompt and filter fixes."""
    _, report, _ = clean
    bad = [f.title for f in report.findings if f.category in ("hardcoded-secret", "sql-injection")]
    assert not bad, bad


def test_overall_precision(vulnerable, clean):
    sample, report, _ = vulnerable
    real = score_findings(sample, report.findings).true_positives
    reported = len(report.findings) + len(clean[1].findings)
    assert real / reported >= MIN_PRECISION, f"{real} real of {reported} reported"


def test_review_all_false_reviews_fewer_units_but_keeps_rule_findings(new_llm, vulnerable):
    _, full, _ = vulnerable
    _, risky, _ = analyze_sample(new_llm, "vulnerable_app.py", VULNERABLE_EXPECTED, review_all=False)
    assert len(risky.reviewed) <= len(full.reviewed)
    assert {f.category for f in risky.findings} >= {"hardcoded-secret", "command-injection"}
