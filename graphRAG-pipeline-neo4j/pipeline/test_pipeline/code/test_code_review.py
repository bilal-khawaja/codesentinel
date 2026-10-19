"""Test of the code review: the real analyzer with the real LLM on the two files in samples/.

    samples/vulnerable_app.py   FastAPI app with 10 planted vulnerabilities and bugs (labels: VULNERABLE_EXPECTED)
    samples/clean_app.py        the same API written correctly, nothing planted

    test_vulnerable_file   the review must report the problems in vulnerable_app.py
    test_clean_file        the review must not report problems in clean_app.py

Run with -s (`just test-review` does) to see each file's findings and its LLM time, tokens and cost.
About $0.03 per session; skipped when no model is configured. The model is not deterministic, so the
thresholds sit just outside what repeated live runs measured: they catch regressions, not one unlucky answer."""
import asyncio
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pipeline.code_analysis.analyzer import analyze_file
from pipeline.code_analysis.evaluation import VULNERABLE_EXPECTED, Sample, score_findings
from pipeline.code_parsing.extractor import extract
from pipeline.code_parsing.language_spec import PYTHON
from pipeline.graphdb.db import get_value_from_env
from pipeline.shared.constants import DEFAULT_LLM_MODEL
from pipeline.llm_setup.llm import get_llm

SAMPLES = Path(__file__).resolve().parent / "samples"
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
MODEL = DEFAULT_LLM_MODEL

MIN_RECALL = 0.8          # of the 10 planted problems
MAX_CLEAN_FINDINGS = 8    # on the clean file; measured 0-7 doubtful findings


def review(name):
    """Review samples/<name> with the real model. Returns the source and the report."""
    if not get_value_from_env(f"LLM_MODEL_CONFIG_{MODEL.upper()}"):
        pytest.skip(f"no LLM configured for {MODEL}")
    source = (SAMPLES / name).read_text(encoding="utf-8")
    llm = get_llm(MODEL)[0]
    report = asyncio.run(analyze_file(llm, source, name, extract(source, PYTHON)))
    print(f"\n--- LLM metrics: {name} ---\n{llm.metrics.report()}")
    return source, report


def print_summary(name, report, score=None):
    """The review of one file in plain text: verdict, every finding with its fix, and what the scoring says."""
    findings = report.findings
    counts = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    by_severity = ", ".join(f"{n} {severity}" for severity, n in counts.items()) or "none"
    line = "=" * 78
    print(f"\n{line}")
    print(f"REVIEW OF {name}: {len(findings)} findings ({by_severity}) | risk: {report.summary.risk_level}")
    print(f"Summary: {report.summary.text}")
    for f in findings:
        print(f"\n  line {f.start_line:<4} [{f.severity}] {f.category} in {f.symbol or '<module>'} ({f.detector})")
        print(f"    {f.title}")
        print(f"    evidence: {f.evidence.strip()}")
        print(f"    fix: {f.fix}")
    if score is not None:
        print(f"\n  planted problems found: {score.true_positives} of {score.true_positives + score.false_negatives}"
              f" | missed: {score.missed or 'none'} | not planted: {score.extra or 'none'}")
    print(f"  dismissed by the reviewer: {len(report.dismissed)} | dropped by verification: {len(report.dropped)}"
          f" | errors: {report.errors or 'none'}")
    print(line)


def test_vulnerable_file():
    """The file with errors: the review must find them."""
    source, report = review("vulnerable_app.py")
    score = score_findings(Sample("vulnerable_app.py", source, VULNERABLE_EXPECTED), report.findings)
    print_summary("vulnerable_app.py", report, score)
    assert not report.errors, report.errors
    assert report.findings, "no findings reported for a file full of problems"
    assert score.recall >= MIN_RECALL, f"missed: {score.missed}"


def test_clean_file():
    """The correct file: the review must not report errors in it."""
    _, report = review("clean_app.py")
    print_summary("clean_app.py", report)
    assert not report.errors, report.errors
    assert len(report.findings) <= MAX_CLEAN_FINDINGS
    bad = [f.title for f in report.findings if f.category in ("hardcoded-secret", "sql-injection")]
    assert not bad, bad
