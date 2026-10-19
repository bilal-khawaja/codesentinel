"""Tests of llm_setup/metrics.py: time, tokens and cost per LLM call. Offline."""
from types import SimpleNamespace

from pipeline.llm_setup.metrics import LLMMetrics, calculate_cost


def usage(input_tokens, output_tokens):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=0)


def test_price_override_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_MY_MODEL_V1", "2.0,4.0")
    assert calculate_cost("my-model-v1", 1_000_000, 500_000) == 4.0


def test_unknown_model_cost_is_none(monkeypatch):
    monkeypatch.delenv("LLM_PRICE_NO_SUCH_MODEL", raising=False)
    assert calculate_cost("no-such-model", 10, 10) is None


def test_summary_adds_up_calls(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_M", "1.0,1.0")
    metrics = LLMMetrics("m")
    metrics.record("a", 0.0, 1.0, usage(100, 50))
    metrics.record("b", 0.5, 3.0, usage(10, 5))
    s = metrics.summary()
    assert (s["calls"], s["input_tokens"], s["output_tokens"], s["total_tokens"]) == (2, 110, 55, 165)
    assert s["wall_seconds"] == 3.0 and s["total_seconds"] == 3.5 and s["slowest"] == "b"
    assert s["cost_complete"] is True


def test_empty_metrics_and_reset():
    metrics = LLMMetrics("m")
    assert metrics.report() == "LLM metrics: no calls"
    metrics.record("a", 0.0, 1.0, usage(1, 1))
    metrics.reset()
    assert metrics.summary()["calls"] == 0
