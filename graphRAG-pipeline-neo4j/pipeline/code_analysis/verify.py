"""Checks on what the LLM says, done in plain code: models invent line numbers and quotes."""
import re
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

