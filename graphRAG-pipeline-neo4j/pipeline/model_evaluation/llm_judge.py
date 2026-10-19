"""deepeval judge backed by the pipeline's own LLM (Groq), so metrics never fall back to OpenAI.

    metric = FaithfulnessMetric(model=LLMJudge(), threshold=0.8)
"""
from typing import Optional

from deepeval.models import DeepEvalBaseLLM

from ..graphdb.db import get_value_from_env
from ..shared.constants import DEFAULT_LLM_MODEL
from ..llm_setup.llm import get_llm

JUDGE_MODEL = get_value_from_env("DEFAULT_TEST_MODEL", DEFAULT_LLM_MODEL)


class LLMJudge(DeepEvalBaseLLM):
    def __init__(self, model_key: Optional[str] = None):
        self.llm, model_name, _ = get_llm(model_key or JUDGE_MODEL)
        super().__init__(model_name)  # also calls load_model()

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.llm.structured("", prompt, schema, label="judge")
        return self.llm.complete("", prompt, label="judge")

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self.llm.astructured("", prompt, schema, label="judge")
        return await self.llm.acomplete("", prompt, label="judge")

    def get_model_name(self) -> str:
        return self.name
