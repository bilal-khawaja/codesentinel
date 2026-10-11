import asyncio
import logging
from typing import Dict, List

from pydantic import BaseModel

from ..shared.constants import CODE_SUMMARY_SYSTEM_PROMPT
from .chunker import MODULE_SYMBOL
from .extractor import CodeInfo

BATCH_SIZE = 8
MAX_SNIPPET_CHARS = 3000


class _Summary(BaseModel):
    id: str
    summary: str


class _Summaries(BaseModel):
    summaries: List[_Summary]


def build_summary_inputs(info: CodeInfo) -> Dict[str, str]:
    """Code snippets that need an LLM summary: every symbol without a docstring,
    plus the module itself when it has none (summarized from its imports and signatures)."""
    inputs = {s.qualname: s.source[:MAX_SNIPPET_CHARS] for s in info.symbols if s.kind != "Variable" and not s.docstring}
    top_level = [s for s in info.symbols if s.parent is None]
    if not info.module_docstring and top_level:
        overview = "imports: " + ", ".join(e.split("|")[1] for e in info.imports) + "\n" + "\n".join(s.signature for s in top_level)
        inputs[MODULE_SYMBOL] = overview[:MAX_SNIPPET_CHARS]
    return inputs


async def summarize_symbols(llm, inputs: Dict[str, str], max_concurrency: int = 5) -> Dict[str, str]:
    """One-sentence summaries for {id: code}. Batches several snippets per LLM call.
    Summaries are a bonus, so a failed batch is logged and skipped, not fatal."""
    items = list(inputs.items())
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(batch) -> Dict[str, str]:
        prompt = "\n\n".join(f"### id: {item_id}\n{code}" for item_id, code in batch)
        async with semaphore:
            try:
                result = await llm.astructured(CODE_SUMMARY_SYSTEM_PROMPT, prompt, _Summaries)
            except Exception as e:
                logging.error(f"Code summary batch failed: {e}")
                return {}
        return {s.id: s.summary.strip() for s in result.summaries if s.id in inputs}

    summaries = {}
    for batch_result in await asyncio.gather(*(run(batch) for batch in batches)):
        summaries.update(batch_result)
    return summaries
