"""Analysis of one source file: rule scan, then LLM review where it pays off, then a summary."""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from pydantic import BaseModel

from ..code_parsing.extractor import CodeInfo
from ..shared.constants import CODE_FILE_SUMMARY_SYSTEM_PROMPT
from .context import ReviewUnit, SymbolContext, build_units, findings_in_unit, needs_review
from .models import SEVERITY_ORDER, Finding
from .reviewer import DismissedFinding, review_unit
from .rules import scan_source
from .verify import redact

# Called with the unit to review, returns where it sits in the call graph (None = no graph).
ContextProvider = Callable[[ReviewUnit], SymbolContext]


class _SummaryText(BaseModel):
    summary: str


@dataclass
class FileSummary:
    risk_level: str   # highest severity among the findings, or "none"
    text: str


@dataclass
class AnalysisReport:
    file: str
    findings: List[Finding] = field(default_factory=list)
    dismissed: List[DismissedFinding] = field(default_factory=list)  # rule findings the LLM judged false positives
    reviewed: List[str] = field(default_factory=list)                # units sent to the LLM
    dropped: List[str] = field(default_factory=list)                 # LLM findings that failed verification
    errors: List[str] = field(default_factory=list)
    summary: Optional[FileSummary] = None
    llm_metrics: Optional[dict] = None                               # time, tokens, cost of this file's LLM calls


async def analyze_file(
    llm,
    source: str,
    file_name: str,
    info: CodeInfo,
    get_context: Optional[ContextProvider] = None,
    max_concurrency: int = 5,
    review_all: bool = True,
) -> AnalysisReport:
    """Rules first (free and exact), then the LLM reviews the code, confirming or dismissing
    rule hits and adding what rules cannot see. review_all=True reviews every function, which
    is what finds logic bugs in small functions (about $0.001 each on a mini model).
    review_all=False reviews only flagged, risky, reachable or long units, to save cost on big projects."""
    report = AnalysisReport(file=file_name)
    rule_findings = scan_source(source, file_name, info)

    if any(f.rule_id == "syntax-error" for f in rule_findings):
        # the file does not parse, so its structure is unreliable: report that and stop
        report.findings = rule_findings
        report.errors.append("syntax error: skipped LLM review")
        report.summary = await summarize_findings(llm, file_name, report.findings, 0)
        report.llm_metrics = _metrics_of(llm)
        return report

    units = build_units(source, info, file_name)
    semaphore = asyncio.Semaphore(max_concurrency)
    unit_findings = {id(unit): findings_in_unit(unit, rule_findings) for unit in units}
    contexts = {id(unit): (get_context(unit) if get_context else SymbolContext()) for unit in units}
    selected = [u for u in units if review_all or needs_review(u, unit_findings[id(u)], contexts[id(u)])]

    async def run(unit: ReviewUnit):
        async with semaphore:
            return await review_unit(llm, unit, unit_findings[id(unit)], contexts[id(unit)], info)

    results = await asyncio.gather(*(run(u) for u in selected))

    reviewed_ids = set()
    for result in results:
        report.findings += result.findings
        report.dismissed += result.dismissed
        report.dropped += result.dropped
        report.reviewed.append(result.unit.symbol or "<module>")
        if result.error:
            report.errors.append(f"{result.unit.symbol or '<module>'}: {result.error}")
        reviewed_ids |= {f.id for f in unit_findings[id(result.unit)]}

    # rule findings in code that was not sent to the LLM stay as they are
    report.findings += [f for f in rule_findings if f.id not in reviewed_ids]
    report.findings = sorted({f.id: f for f in report.findings}.values(), key=lambda f: (f.start_line, f.id))

    report.summary = await summarize_findings(llm, file_name, report.findings, len(selected))
    report.llm_metrics = _metrics_of(llm)
    return report


def risk_level(findings: List[Finding]) -> str:
    levels = [f.severity for f in findings]
    return next((s for s in SEVERITY_ORDER if s in levels), "none")


async def summarize_findings(llm, file_name: str, findings: List[Finding], reviewed_units: int) -> FileSummary:
    """Plain-language summary of a file's findings. The risk level is computed from the
    findings, never taken from the LLM. With no findings no LLM call is made."""
    level = risk_level(findings)
    if not findings:
        return FileSummary("none", f"No vulnerabilities or bugs were detected in {file_name} "
                                   f"({reviewed_units} units reviewed by the LLM, rule scan clean). "
                                   "This is not proof the code is safe.")
    listing = "\n".join(
        f"- {f.severity} | {f.kind} | {f.title} | in {f.symbol or 'module level'} line {f.start_line}" for f in findings
    )
    try:
        output = await llm.astructured(
            CODE_FILE_SUMMARY_SYSTEM_PROMPT, f"File: {file_name}\nFindings:\n{listing}", _SummaryText,
            label="summary:" + file_name,
        )
        return FileSummary(level, redact(output.summary.strip()))
    except Exception as e:
        logging.error(f"File summary failed for {file_name}: {e}")
        counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        return FileSummary(level, f"{len(findings)} findings in {file_name} (" +
                           ", ".join(f"{n} {s}" for s, n in counts.items() if n) + ").")


def _metrics_of(llm) -> Optional[dict]:
    metrics = getattr(llm, "metrics", None)
    return metrics.summary() if metrics else None
