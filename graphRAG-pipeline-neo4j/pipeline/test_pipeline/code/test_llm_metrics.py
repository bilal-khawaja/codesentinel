"""Tests for llm_setup/metrics.py and its use by LLMHandle. Offline: pydantic-ai's TestModel stands in for OpenAI."""
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pipeline.graphdb.db import UniversalTokenUsageHandler
from pipeline.llm_setup.llm import LLMHandle
from pipeline.llm_setup.metrics import LLMMetrics, calculate_cost


def usage(input_tokens, output_tokens):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=0)


def test_cost_comes_from_the_price_database():
    assert calculate_cost("gpt-4.1-mini", 1_000_000, 1_000_000) == pytest.approx(2.0)  # $0.40 in + $1.60 out
    assert calculate_cost("gpt-4.1-mini", 500_000, 0) == pytest.approx(0.2)


def test_unknown_model_cost_is_unknown_not_guessed():
    assert calculate_cost("my-private-model", 1000, 1000) is None


def test_price_override_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_MY_PRIVATE_MODEL", "1.0,4.0")
    assert calculate_cost("my-private-model", 1_000_000, 500_000) == pytest.approx(3.0)


def test_malformed_override_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_MY_PRIVATE_MODEL", "cheap")
    assert calculate_cost("my-private-model", 1000, 1000) is None


def test_summary_adds_up_calls():
    metrics = LLMMetrics("gpt-4.1-mini")
    metrics.record("a", started=0.0, ended=2.0, usage=usage(1000, 100))
    metrics.record("b", started=1.0, ended=4.0, usage=usage(2000, 300))   # overlaps the first call
    summary = metrics.summary()
    assert summary["calls"] == 2
    assert (summary["input_tokens"], summary["output_tokens"], summary["total_tokens"]) == (3000, 400, 3400)
    assert summary["total_seconds"] == 5.0 and summary["wall_seconds"] == 4.0 and summary["average_seconds"] == 2.5
    assert summary["slowest"] == "b"
    assert summary["cost_usd"] == pytest.approx(3000 * 0.4 / 1e6 + 400 * 1.6 / 1e6, rel=1e-3)
    assert summary["cost_complete"] is True


def test_unknown_model_summary_has_no_cost():
    metrics = LLMMetrics("my-private-model")
    metrics.record("a", 0.0, 1.0, usage(10, 10))
    assert metrics.summary()["cost_usd"] is None and metrics.summary()["cost_complete"] is False
    assert "cost unknown" in metrics.report()


def test_empty_metrics_and_reset():
    metrics = LLMMetrics("gpt-4.1-mini")
    assert metrics.summary()["calls"] == 0 and "no calls" in metrics.report()
    metrics.record("a", 0.0, 1.0, usage(1, 1))
    metrics.reset()
    assert metrics.summary()["calls"] == 0


def test_failed_call_without_usage_is_still_timed():
    metrics = LLMMetrics("gpt-4.1-mini")
    call = metrics.record("a", 0.0, 0.5, usage=None)
    assert call.total_tokens == 0 and call.seconds == 0.5


class Answer(BaseModel):
    text: str


async def test_llm_handle_records_every_call():
    llm = LLMHandle(TestModel(), "gpt-4.1-mini", UniversalTokenUsageHandler())
    await llm.astructured("system", "user prompt", Answer, label="first")
    await llm.acomplete("system", "user prompt", label="second")
    assert [c.label for c in llm.metrics.calls] == ["first", "second"]
    assert all(c.seconds > 0 for c in llm.metrics.calls)
    assert "2 calls" in llm.metrics.report()


def test_llm_handle_feeds_tokens_to_metrics_and_the_older_counter():
    """TestModel reports no token usage, so hand the handle a result that does."""
    llm = LLMHandle(TestModel(), "gpt-4.1-mini", UniversalTokenUsageHandler())
    result = SimpleNamespace(usage=lambda: usage(1200, 300))
    llm._record(result, "review:f", started=time.perf_counter() - 1.5)
    call = llm.metrics.calls[0]
    assert (call.input_tokens, call.output_tokens, call.label) == (1200, 300, "review:f")
    assert call.seconds >= 1.5 and call.cost_usd == pytest.approx(1200 * 0.4 / 1e6 + 300 * 1.6 / 1e6)
    assert llm.usage_tracker.report()["total_tokens"] == 1500 == llm.metrics.summary()["total_tokens"]


def test_usage_can_be_a_property_or_a_method():
    """pydantic-ai 2.x exposes result.usage as a property; older versions as a method."""
    llm = LLMHandle(TestModel(), "gpt-4.1-mini", UniversalTokenUsageHandler())
    llm._record(SimpleNamespace(usage=usage(10, 5)), "property", started=time.perf_counter())
    llm._record(SimpleNamespace(usage=lambda: usage(10, 5)), "method", started=time.perf_counter())
    assert [c.total_tokens for c in llm.metrics.calls] == [15, 15]


def test_run_agent_records_the_whole_agent_run():
    llm = LLMHandle(TestModel(), "gpt-4.1-mini", UniversalTokenUsageHandler())
    agent = Agent(TestModel(), output_type=str)
    llm.run_agent(agent, "hello", label="query")
    assert [c.label for c in llm.metrics.calls] == ["query"]
