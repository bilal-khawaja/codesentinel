import hashlib
from typing import Literal, Optional

from pydantic import BaseModel

Kind = Literal["vulnerability", "bug"]
Severity = Literal["critical", "high", "medium", "low", "info"]
Confidence = Literal["high", "medium", "low"]
Detector = Literal["rule", "llm", "rule+llm"]  # rule+llm: found by a rule, confirmed and explained by the LLM

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


class Finding(BaseModel):
    """One vulnerability or bug in one place in the code. Rules and the LLM reviewer
    both produce this, so everything downstream (storage, retrieval) handles one shape."""
    id: str
    kind: Kind
    category: str                 # e.g. "sql-injection", "syntax-error"
    cwe: Optional[str] = None     # e.g. "CWE-89"
    severity: Severity
    confidence: Confidence
    title: str
    explanation: str
    scenario: str                 # how it is exploited or how it fails at runtime
    fix: str
    file: str
    symbol: Optional[str] = None  # qualname of the function/class it is in, None = module level
    start_line: int
    end_line: int
    evidence: str                 # the offending line (secrets are masked)
    detector: Detector
    rule_id: Optional[str] = None


def make_finding_id(file: str, symbol: Optional[str], rule_id: str, start_line: int) -> str:
    """Stable, short id so a finding can be quoted in a chat ("explain F-3a9c1d0b2e")."""
    digest = hashlib.sha1(f"{file}|{symbol}|{rule_id}|{start_line}".encode("utf-8")).hexdigest()
    return f"F-{digest[:10]}"
