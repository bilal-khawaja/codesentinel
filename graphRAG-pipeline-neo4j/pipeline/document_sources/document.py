from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """Drop-in replacement for langchain_core.documents.Document — a page of
    text plus metadata. Only the two attributes downstream code touches
    (page_content, metadata) are provided."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
