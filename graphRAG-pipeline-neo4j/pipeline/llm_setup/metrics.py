"""Time, tokens and cost of LLM calls.

Every LLMHandle owns one LLMMetrics and records each call into it, so after any piece of work:

    llm, _, _ = get_llm("openai_gpt_4_1_mini")
    ... run work ...
    print(llm.metrics.report())      # readable text
    llm.metrics.summary()            # dict, for logs or API responses
    llm.metrics.calls                # one CallMetrics per call

Prices come from the `genai-prices` database that ships with pydantic-ai. For a model it does
not know (a custom endpoint), set LLM_PRICE_<MODEL>="<input>,<output>" in USD per million
tokens, e.g. LLM_PRICE_MY_MODEL_V1="0.5,2.0". Without either, cost is reported as unknown
instead of guessed."""
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import List, Optional

from genai_prices import Usage, calc_price


@dataclass
class CallMetrics:
    label: str                   # what the call was for, e.g. "review:Gateway.charge"
    model: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]    # None when the model's price is unknown

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Optional[float]:
    """USD for one call, or None if the price of the model is unknown."""
    override = os.environ.get("LLM_PRICE_" + re.sub(r"[^A-Z0-9]+", "_", model_name.upper()))
    if override:
        try:
            per_million_in, per_million_out = (float(part) for part in override.split(","))
            return (input_tokens * per_million_in + output_tokens * per_million_out) / 1_000_000
        except ValueError:
            logging.error(f"Ignoring malformed price override for {model_name}: {override!r}")
    try:
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cached_tokens or None)
        return float(calc_price(usage, model_ref=model_name).total_price)
    except Exception:  # unknown model, or pricing data without that model
        return None


class LLMMetrics:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: List[CallMetrics] = []
        self._first_start: Optional[float] = None
        self._last_end: Optional[float] = None
        self._lock = threading.Lock()  # calls are recorded from concurrent tasks and threads

    def record(self, label: str, started: float, ended: float, usage=None) -> CallMetrics:
        """Add one finished call. `started` and `ended` are time.perf_counter() values and
        `usage` is the pydantic-ai usage of the run (may be None if the call failed)."""
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cached_tokens = getattr(usage, "cache_read_tokens", 0) or 0
        call = CallMetrics(
            label=label or "call",
            model=self.model_name,
            seconds=ended - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(self.model_name, input_tokens, output_tokens, cached_tokens),
        )
        with self._lock:
            self.calls.append(call)
            self._first_start = started if self._first_start is None else min(self._first_start, started)
            self._last_end = ended if self._last_end is None else max(self._last_end, ended)
        return call

    def reset(self):
        with self._lock:
            self.calls.clear()
            self._first_start = self._last_end = None

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
            wall = (self._last_end - self._first_start) if calls else 0.0
        known = [c.cost_usd for c in calls if c.cost_usd is not None]
        return {
            "model": self.model_name,
            "calls": len(calls),
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "total_tokens": sum(c.total_tokens for c in calls),
            "cost_usd": round(sum(known), 6) if known else None,
            "cost_complete": len(known) == len(calls),   # False: some calls had no known price
            "total_seconds": round(sum(c.seconds for c in calls), 3),  # added up, so more than wall time when calls overlap
            "wall_seconds": round(wall, 3),
            "average_seconds": round(sum(c.seconds for c in calls) / len(calls), 3) if calls else 0.0,
            "slowest": max(calls, key=lambda c: c.seconds).label if calls else None,
        }

    def report(self) -> str:
        s = self.summary()
        if not s["calls"]:
            return "LLM metrics: no calls"
        cost = "unknown" if s["cost_usd"] is None else f"${s['cost_usd']:.4f}" + ("" if s["cost_complete"] else " (partial)")
        return (
            f"LLM metrics ({s['model']}): {s['calls']} calls | "
            f"time {s['wall_seconds']}s wall, {s['total_seconds']}s summed, {s['average_seconds']}s avg | "
            f"tokens {s['total_tokens']} ({s['input_tokens']} in, {s['output_tokens']} out) | cost {cost}"
        )
