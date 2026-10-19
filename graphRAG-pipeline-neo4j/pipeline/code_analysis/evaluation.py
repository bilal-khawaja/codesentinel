"""Measures the analyzer: how many planted problems it finds (recall) and how many of its
findings are real (precision), on files whose problems are known.

A sample is a source file plus the problems planted in it (`Expected`). Each expected problem
names a family ("sql-injection"), and an `anchor`: text on the offending line, so the label
follows the code when the sample is edited and the model never sees the answer as a comment.
A finding matches an expected problem when it is in the same family and touches its line
(within `tolerance` lines). Every finding that matches nothing counts as a false positive,
so on a clean sample every finding is one.

Run on the bundled samples (real LLM, costs a few cents):
    uv run python -m pipeline.code_analysis.evaluation --runs 3"""
import argparse
import asyncio
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..code_parsing.extractor import extract
from ..code_parsing.language_spec import PYTHON
from .analyzer import analyze_file
from .models import SEVERITY_ORDER, Finding
from .rules import scan_source
from ..shared.constants import DEFAULT_LLM_MODEL

# first match wins; matched against category, title and cwe of a finding
FAMILIES = [
    ("sql-injection", r"sql|cypher|cql|cwe-89\b"),
    ("command-injection", r"command|shell|os-command|subprocess|cwe-78\b"),
    ("code-injection", r"\beval|\bexec|code-injection|cwe-95\b"),
    ("deserialization", r"deserial|pickle|yaml|marshal|cwe-502\b"),
    ("tls", r"tls|ssl|certificate|verify|cwe-295\b"),
    ("weak-crypto", r"weak|md5|sha-?1|crypto|hash|cwe-32[78]\b|cwe-916\b"),
    ("hardcoded-secret", r"secret|credential|password|api-?key|token|hardcoded|cwe-798\b"),
    ("mutable-default", r"mutable|cwe-1188\b"),
    ("path-traversal", r"path|traversal|cwe-22\b"),
    ("index-error", r"index|off-by-one|bounds|out-of-range|range"),
    ("none-dereference", r"none|null|attribute|dereference|cwe-476\b"),
]


def family_of(finding: Finding) -> str:
    text = f"{finding.category} {finding.title} {finding.cwe or ''}".lower()
    for family, pattern in FAMILIES:
        if re.search(pattern, text):
            return family
    return "other"


@dataclass
class Expected:
    family: str
    anchor: str            # text that appears on the offending line, once in the file
    tolerance: int = 2     # lines either side that still count as the same place


@dataclass
class Sample:
    name: str
    source: str
    expected: List[Expected] = field(default_factory=list)   # empty = a clean file


@dataclass
class Score:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    missed: List[str] = field(default_factory=list)          # expected families nobody found
    extra: List[str] = field(default_factory=list)           # findings that matched nothing

    @property
    def precision(self) -> Optional[float]:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else None

    @property
    def recall(self) -> Optional[float]:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else None

    def add(self, other: "Score") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives
        self.missed += other.missed
        self.extra += other.extra


def anchor_line(source: str, anchor: str) -> int:
    lines = [n for n, text in enumerate(source.splitlines(), 1) if anchor in text]
    if len(lines) != 1:
        raise ValueError(f"anchor {anchor!r} must appear on exactly one line, found {len(lines)}")
    return lines[0]


def score_findings(sample: Sample, findings: List[Finding]) -> Score:
    """One finding can satisfy at most one expected problem, and the other way round."""
    score = Score()
    unmatched = list(findings)
    for expected in sample.expected:
        line = anchor_line(sample.source, expected.anchor)
        hit = next((f for f in unmatched if family_of(f) == expected.family
                    and f.start_line - expected.tolerance <= line <= f.end_line + expected.tolerance), None)
        if hit:
            unmatched.remove(hit)
            score.true_positives += 1
        else:
            score.false_negatives += 1
            score.missed.append(f"{expected.family} @ line {line}")
    score.false_positives = len(unmatched)
    score.extra = [f"{f.severity} {family_of(f)} @ line {f.start_line}: {f.title}" for f in unmatched]
    return score


def at_least(findings: List[Finding], severity: str) -> List[Finding]:
    keep = SEVERITY_ORDER[:SEVERITY_ORDER.index(severity) + 1]
    return [f for f in findings if f.severity in keep]


@dataclass
class SampleResult:
    sample: Sample
    findings: List[Finding]
    score: Score
    errors: List[str] = field(default_factory=list)
    dropped: int = 0
    llm_metrics: Optional[dict] = None


@dataclass
class EvalReport:
    results: List[SampleResult] = field(default_factory=list)

    @property
    def total(self) -> Score:
        total = Score()
        for result in self.results:
            total.add(result.score)
        return total

    @property
    def cost_usd(self) -> float:
        return sum((r.llm_metrics or {}).get("cost_usd", 0) or 0 for r in self.results)

    def text(self) -> str:
        lines = []
        for r in self.results:
            s = r.score
            lines.append(f"{r.sample.name}: {len(r.findings)} findings | TP {s.true_positives} FP {s.false_positives} "
                         f"FN {s.false_negatives} | precision {_pct(s.precision)} recall {_pct(s.recall)}")
            lines += [f"    missed: {m}" for m in s.missed] + [f"    extra:  {e}" for e in s.extra]
            lines += [f"    ERROR:  {e}" for e in r.errors]
            if r.dropped:
                lines.append(f"    dropped by verification: {r.dropped}")
        t = self.total
        lines.append(f"TOTAL: TP {t.true_positives} FP {t.false_positives} FN {t.false_negatives} | "
                     f"precision {_pct(t.precision)} recall {_pct(t.recall)} | cost ${self.cost_usd:.4f}")
        return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


async def evaluate(llm, samples: List[Sample], use_llm: bool = True, review_all: bool = True,
                   min_severity: str = "info") -> EvalReport:
    """Analyze every sample the way ingest does (rules, then LLM review) and score the result.
    use_llm=False scores the rule scan alone, the baseline the LLM has to beat."""
    report = EvalReport()
    for sample in samples:
        info = extract(sample.source, PYTHON)
        if use_llm:
            analysis = await analyze_file(llm, sample.source, sample.name, info, review_all=review_all)
            findings, errors, dropped, metrics = analysis.findings, analysis.errors, len(analysis.dropped), analysis.llm_metrics
        else:
            findings, errors, dropped, metrics = scan_source(sample.source, sample.name, info), [], 0, None
        findings = at_least(findings, min_severity)
        report.results.append(SampleResult(sample, findings, score_findings(sample, findings), errors, dropped, metrics))
    return report


def summarize_runs(reports: List[EvalReport]) -> str:
    """Several runs of the same samples: the model is not deterministic, so show the spread."""
    recalls = [r.total.recall for r in reports if r.total.recall is not None]
    precisions = [r.total.precision for r in reports if r.total.precision is not None]
    findings = [sum(len(x.findings) for x in r.results) for r in reports]
    parts = [f"{len(reports)} runs"]
    for name, values in (("recall", recalls), ("precision", precisions)):
        if values:
            parts.append(f"{name} mean {statistics.mean(values):.0%} (min {min(values):.0%}, max {max(values):.0%})")
    parts.append(f"findings per run {min(findings)}-{max(findings)}")
    parts.append(f"total cost ${sum(r.cost_usd for r in reports):.4f}")
    return " | ".join(parts)


# --- the bundled samples ---

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_pipeline" / "code" / "samples"

VULNERABLE_EXPECTED = [
    Expected("sql-injection", "cur.execute(f\"SELECT"),
    Expected("command-injection", "os.system(\"ping"),
    Expected("deserialization", "pickle.loads(blob)"),
    Expected("hardcoded-secret", "PAYMENT_API_KEY ="),
    Expected("tls", "verify=False"),
    Expected("mutable-default", "def add_item(item, items=[])"),
    Expected("index-error", "range(len(prices) + 1)"),
    Expected("none-dereference", "match.group(1)"),
    Expected("weak-crypto", "hashlib.md5"),
    Expected("path-traversal", "os.path.join(BASE_DIR, name)"),
]


def load_bundled_samples() -> List[Sample]:
    """vulnerable_app.py with its planted problems, and clean_app.py with none."""
    vulnerable = (SAMPLES_DIR / "vulnerable_app.py").read_text(encoding="utf-8")
    clean = (SAMPLES_DIR / "clean_app.py").read_text(encoding="utf-8")
    return [Sample("vulnerable_app.py", vulnerable, VULNERABLE_EXPECTED), Sample("clean_app.py", clean, [])]


def main() -> None:
    from dotenv import load_dotenv
    from ..llm_setup.llm import get_llm

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # [OPENAI-DISABLED] parser.add_argument("--model", default="openai_gpt_4_1_mini")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--rules-only", action="store_true", help="score the rule scan without any LLM call")
    parser.add_argument("--risky-only", action="store_true", help="LLM reviews only flagged or risky units (review_all=False)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    samples = load_bundled_samples()

    async def run_once() -> EvalReport:
        llm = None if args.rules_only else get_llm(args.model)[0]
        return await evaluate(llm, samples, use_llm=not args.rules_only, review_all=not args.risky_only)

    reports = []
    for number in range(1, args.runs + 1):
        report = asyncio.run(run_once())
        reports.append(report)
        print(f"--- run {number} ---\n{report.text()}")
    if len(reports) > 1:
        print("\n" + summarize_runs(reports))


if __name__ == "__main__":
    main()
