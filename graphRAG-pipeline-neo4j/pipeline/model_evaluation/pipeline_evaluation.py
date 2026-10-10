from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv
from ..retrievals.retrieve import retrieve,get_neo4j_credentials
from ..graphdb.db import get_value_from_env
from pathlib import Path    

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PACKAGE_ROOT / ".env"
if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Environment file not found: {ENV_FILE}")

load_dotenv(dotenv_path=ENV_FILE, override=False)
if not all(get_value_from_env(key) for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")):
    raise RuntimeError(f"Required Neo4j settings were not loaded from {ENV_FILE}")

MODEL = get_value_from_env('DEFAULT_TEST_MODEL', 'openai_gpt_4_1_mini')
TEST_DATA_DIR = PACKAGE_ROOT / "test_data"
SAMPLE_FILE = TEST_DATA_DIR / "Apple stock during pandemic.pdf"

def model_eval():
    question = (
    "What was the lowest recorded closing price of Apple's stock "
    "during the pandemic, and on what date did it occur?"
    )

    expected_answer = (
        "Apple's lowest recorded closing price during the pandemic was "
        "$56.09, occurring on March 23, 2020."
    )

    creds = get_neo4j_credentials()

    result = retrieve(
        query=question,
        mode="graph_vector",
        credentials=creds,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        model=MODEL,
    )
    print(f"Retrieve result: {result}")
    test_case = LLMTestCase(
        input=question,
        expected_output=expected_answer,
        actual_output=result["answer"],
        retrieval_context =result['retrieval_context'],
    )

    # faithfulness = FaithfulnessMetric(
    #     threshold=0.8,
    #     include_reason = True,
    # )

    # answer_relevancy = AnswerRelevancyMetric(
    #     threshold=0.8,
    #     include_reason = True,
    # )
    # faithfulness.measure(test_case)
    # answer_relevancy.measure(test_case)

    # print("Faithfulness:")
    # print("Score:", faithfulness.score)
    # print("Reason:", faithfulness.reason)

    # print("\nAnswer Relevancy:")
    # print("Score:", answer_relevancy.score)
    # print("Reason:", answer_relevancy.reason)

    # assert_test(
    #     test_case,
    #     [
    #         faithfulness,
    #         answer_relevancy,
    #     ]
    # )

    answer_correctness = GEval(
    name="Answer Correctness",
    # criteria is the instructions given to the LLM evaluator to determine whether the actual answer is correct and agrees with the expected answer.
    criteria=(
        "Determine whether the actual answer correctly answers the input "
        "question and agrees with the expected answer. "
        "Pay particular attention to factual details such as numbers, "
        "dates, names, and other specific values. "
        "Penalize missing or incorrect factual details."
    ),
    # evaluation_params specifies which parameters of the test case will be used for evaluation, taken from LLMTestCase.
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.8,
)
    answer_correctness.measure(test_case)

    print("\n" + "=" * 60)
    print("ANSWER CORRECTNESS")
    print("=" * 60)
    print("Score:", answer_correctness.score)
    print("Reason:", answer_correctness.reason)

if __name__ == "__main__":
    model_eval()

