"""Eval: "who calls X" must be answered by graph traversal (get_neighbors), not text search.

Needs a Neo4j with this file already ingested (ingest(<path to this file's sample>) first):

    def helper(x): ...
    class Shape:
        def area(self): return helper(1)
    class Square(Shape):
        def area(self): return helper(self.side)
"""
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ..graphdb.db import get_value_from_env
from ..llm_setup.llm import get_llm

MODEL = get_value_from_env("DEFAULT_TEST_MODEL", "openai_gpt_4_1_mini")


def eval_who_calls_helper():
    question = "Which functions call helper?"
    llm, _, _ = get_llm(MODEL)
    answer = llm.create_query_agent().run_sync(question).output

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output="Shape.area and Square.area call helper.",
    )
    metric = GEval(
        name="Callers correct",
        criteria="The answer names exactly Shape.area and Square.area as the callers of helper.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
    )
    assert_test(test_case, [metric])
