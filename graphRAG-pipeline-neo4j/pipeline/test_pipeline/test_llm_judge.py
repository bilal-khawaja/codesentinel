"""Tests of model_evaluation/llm_judge.py: the deepeval judge that routes through the pipeline's own LLM, so
metrics never fall back to OpenAI. Offline: pydantic-ai's TestModel stands in for Groq."""
import asyncio

import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval, HallucinationMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from pipeline.graphdb.db import UniversalTokenUsageHandler
from pipeline.llm_setup.llm import LLMHandle
from pipeline.model_evaluation import llm_judge
from pipeline.model_evaluation.llm_judge import LLMJudge

MODEL_NAME = "openai/gpt-oss-20b"


class Verdict(BaseModel):
    verdict: str


@pytest.fixture
def requested_keys(monkeypatch):
    """Replace get_llm with an offline handle; returns the list of model keys it was asked for."""
    keys = []

    def fake_get_llm(key):
        keys.append(key)
        model = TestModel(profile=ModelProfile(supports_json_schema_output=True), custom_output_text='{"verdict": "ok"}')
        return LLMHandle(model, MODEL_NAME, UniversalTokenUsageHandler()), MODEL_NAME, None

    monkeypatch.setattr(llm_judge, "get_llm", fake_get_llm)
    return keys


def test_judge_is_a_deepeval_model_named_after_the_llm_model(requested_keys):
    judge = LLMJudge()
    assert isinstance(judge, DeepEvalBaseLLM)
    assert judge.get_model_name() == MODEL_NAME


def test_judge_uses_the_default_judge_model(requested_keys):
    LLMJudge()
    assert requested_keys == [llm_judge.JUDGE_MODEL]
    assert "openai_gpt" not in llm_judge.JUDGE_MODEL


def test_judge_can_use_another_model(requested_keys):
    LLMJudge("groq_other_model")
    assert requested_keys == ["groq_other_model"]


def test_load_model_returns_the_llm_handle(requested_keys):
    judge = LLMJudge()
    assert isinstance(judge.load_model(), LLMHandle)


def test_generate_returns_text(requested_keys):
    assert isinstance(LLMJudge().generate("Is the sky blue?"), str)


def test_generate_with_a_schema_returns_that_schema(requested_keys):
    assert isinstance(LLMJudge().generate("Is the sky blue?", schema=Verdict), Verdict)


def test_a_generate_returns_text(requested_keys):
    assert isinstance(asyncio.run(LLMJudge().a_generate("Is the sky blue?")), str)


def test_a_generate_with_a_schema_returns_that_schema(requested_keys):
    assert isinstance(asyncio.run(LLMJudge().a_generate("Is the sky blue?", schema=Verdict)), Verdict)


def test_judge_calls_are_recorded_as_judge_metrics(requested_keys):
    judge = LLMJudge()
    judge.generate("one")
    judge.generate("two", schema=Verdict)
    assert [call.label for call in judge.llm.metrics.calls] == ["judge", "judge"]


@pytest.mark.parametrize("build", [
    lambda judge: FaithfulnessMetric(model=judge, threshold=0.8),
    lambda judge: AnswerRelevancyMetric(model=judge, threshold=0.8),
    lambda judge: HallucinationMetric(model=judge, threshold=0.8),
    lambda judge: GEval(name="x", model=judge, criteria="c", evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]),
], ids=["faithfulness", "answer_relevancy", "hallucination", "geval"])
def test_deepeval_metrics_accept_the_judge(requested_keys, build):
    judge = LLMJudge()
    assert build(judge).model is judge


def test_eval_files_no_longer_name_an_openai_judge():
    """The eval scripts must pass the Groq judge, not the string "gpt-4.1-mini" (deepeval would call OpenAI)."""
    from pathlib import Path

    eval_dir = Path(llm_judge.__file__).parent
    for path in eval_dir.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if 'model="gpt-4.1-mini"' in line:
                assert line.lstrip().startswith("# [OPENAI-DISABLED]"), f"{path.name}: {line.strip()}"
