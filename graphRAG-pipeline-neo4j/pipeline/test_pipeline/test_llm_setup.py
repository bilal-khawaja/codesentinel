"""Tests of llm_setup/llm.py: building a Gemini or OpenAI-compatible (Groq) model from the env config, the 429 backoff,
and native structured output. Offline: pydantic-ai's FunctionModel stands in for the LLM, time.sleep is patched."""
import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles import ModelProfile

from pipeline.graphdb.db import UniversalTokenUsageHandler
from pipeline.llm_setup import llm as llm_module
from pipeline.llm_setup.llm import LLMHandle, get_llm

CONFIG_KEY = "LLM_MODEL_CONFIG_GROQ_UNIT_TEST"


def http_error(status=429, body=None):
    return ModelHTTPError(status, "openai/gpt-oss-20b", body if body is not None else {"message": "slow down"})


DAILY_QUOTA = {"message": "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199990. Please try again in 1h2m."}


GEMINI_DAILY_QUOTA = {"error": {"message": "Quota exceeded for metric generate_content_free_tier_requests, limit: 20, PerDay"}}


class Verdict(BaseModel):
    verdict: str


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff waits instead of sleeping."""
    waited = []
    monkeypatch.setattr(llm_module.time, "sleep", waited.append)

    async def fake_async_sleep(seconds):
        waited.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_async_sleep)
    return waited


# --- get_llm ---

def test_get_llm_builds_a_groq_model_from_env(monkeypatch):
    monkeypatch.setenv(CONFIG_KEY, "openai/gpt-oss-20b,https://api.groq.com/openai/v1,gsk_test")
    handle, model_name, usage = get_llm("groq_unit_test")
    assert isinstance(handle, LLMHandle)
    assert model_name == "openai/gpt-oss-20b"
    assert handle.model.model_name == "openai/gpt-oss-20b"
    assert "api.groq.com" in str(handle.model.client.base_url)
    assert isinstance(usage, UniversalTokenUsageHandler)


def test_get_llm_key_is_uppercased_with_dots_as_underscores(monkeypatch):
    monkeypatch.setenv(CONFIG_KEY, "m,https://h.example/v1,k")
    assert get_llm("groq.unit.test")[1] == "m"


def test_get_llm_without_config_raises(monkeypatch):
    monkeypatch.delenv(CONFIG_KEY, raising=False)
    with pytest.raises(Exception, match="not defined"):
        get_llm("groq_unit_test")


def test_get_llm_rejects_a_value_without_endpoint(monkeypatch):
    monkeypatch.setenv(CONFIG_KEY, "just-a-model,gsk_test")
    with pytest.raises(Exception, match="Error while creating LLM"):
        get_llm("groq_unit_test")


def test_get_llm_builds_a_gemini_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_CONFIG_GEMINI_UNIT_TEST", "gemini-3.5-flash-lite,test-key")
    handle, model_name, usage = get_llm("gemini_unit_test")
    assert isinstance(handle, LLMHandle)
    assert isinstance(handle.model, GoogleModel)
    assert model_name == "gemini-3.5-flash-lite"
    assert isinstance(usage, UniversalTokenUsageHandler)


def test_get_llm_rejects_a_gemini_value_without_key(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_CONFIG_GEMINI_UNIT_TEST", "gemini-3.5-flash-lite")
    with pytest.raises(Exception, match="Error while creating LLM"):
        get_llm("gemini_unit_test")


# --- backoff delay ---

@pytest.mark.parametrize("attempt,expected", [(0, 2.0), (1, 4.0), (2, 8.0), (4, 32.0)])
def test_rate_limit_delay_doubles(attempt, expected):
    assert llm_module._backoff_delay(http_error(), attempt) == expected


def test_no_delay_after_the_last_retry():
    assert llm_module._backoff_delay(http_error(), llm_module._RATE_LIMIT_MAX_RETRIES) is None


@pytest.mark.parametrize("status", [400, 401, 500, 503])
def test_other_statuses_are_not_retried(status):
    assert llm_module._backoff_delay(http_error(status), 0) is None


def test_daily_quota_is_not_retried():
    assert llm_module._backoff_delay(http_error(429, DAILY_QUOTA), 0) is None
    assert llm_module._backoff_delay(http_error(429, GEMINI_DAILY_QUOTA), 0) is None


# --- backoff loops ---

def test_sync_backoff_retries_until_success(sleeps):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise http_error()
        return "ok"

    assert llm_module._run_sync_with_backoff(flaky) == "ok"
    assert len(calls) == 3 and sleeps == [2.0, 4.0]


def test_sync_backoff_gives_up_and_raises(sleeps):
    def always():
        raise http_error()

    with pytest.raises(ModelHTTPError):
        llm_module._run_sync_with_backoff(always)
    assert len(sleeps) == llm_module._RATE_LIMIT_MAX_RETRIES


@pytest.mark.parametrize("error", [http_error(400), http_error(429, DAILY_QUOTA)])
def test_sync_backoff_raises_immediately_when_retrying_is_pointless(sleeps, error):
    calls = []

    def fails():
        calls.append(1)
        raise error

    with pytest.raises(ModelHTTPError):
        llm_module._run_sync_with_backoff(fails)
    assert len(calls) == 1 and sleeps == []


def test_async_backoff_retries_until_success(sleeps):
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise http_error()
        return "ok"

    assert asyncio.run(llm_module._run_with_backoff(flaky)) == "ok"
    assert len(calls) == 2 and sleeps == [2.0]


# --- LLMHandle ---

def make_handle(respond):
    # Groq's gpt-oss models support json_schema output; the fake model has to say so for NativeOutput
    model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
    return LLMHandle(model, "openai/gpt-oss-20b", UniversalTokenUsageHandler())


def test_structured_uses_native_json_schema_output():
    seen = {}

    def respond(messages, info):
        seen["mode"] = info.model_request_parameters.output_mode
        return ModelResponse(parts=[TextPart('{"verdict": "ok"}')])

    result = make_handle(respond).structured("system", "prompt", Verdict)
    assert result == Verdict(verdict="ok")
    assert seen["mode"] == "native"


def test_astructured_uses_native_json_schema_output():
    seen = {}

    def respond(messages, info):
        seen["mode"] = info.model_request_parameters.output_mode
        return ModelResponse(parts=[TextPart('{"verdict": "ok"}')])

    result = asyncio.run(make_handle(respond).astructured("system", "prompt", Verdict))
    assert result == Verdict(verdict="ok")
    assert seen["mode"] == "native"


def test_complete_survives_a_rate_limit(sleeps):
    calls = []

    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            raise http_error()
        return ModelResponse(parts=[TextPart("fine")])

    assert make_handle(respond).complete("system", "prompt") == "fine"
    assert sleeps == [2.0]


def test_acomplete_survives_a_rate_limit(sleeps):
    calls = []

    def respond(messages, info):
        calls.append(1)
        if len(calls) == 1:
            raise http_error()
        return ModelResponse(parts=[TextPart("fine")])

    assert asyncio.run(make_handle(respond).acomplete("system", "prompt")) == "fine"
    assert sleeps == [2.0]


def test_calls_are_recorded_in_metrics():
    handle = make_handle(lambda messages, info: ModelResponse(parts=[TextPart("hi")]))
    handle.complete("system", "prompt", label="unit")
    assert [call.label for call in handle.metrics.calls] == ["unit"]
