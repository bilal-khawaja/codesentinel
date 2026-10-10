from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from dotenv import load_dotenv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.retrievals.retrieve import retrieve, get_neo4j_credentials
    from pipeline.graphdb.db import get_value_from_env
else:
    from ..retrievals.retrieve import retrieve, get_neo4j_credentials
    from ..graphdb.db import get_value_from_env

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PACKAGE_ROOT / ".env"
if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Environment file not found: {ENV_FILE}")

load_dotenv(dotenv_path=ENV_FILE, override=False)
if not all(get_value_from_env(key) for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")):
    raise RuntimeError(f"Required Neo4j settings were not loaded from {ENV_FILE}")

MODEL = "openai_gpt_4_1_mini"


def test_answer_relevancy():

    question = (
        #"when was ice cream discovered?" # if answer is not in the document and model replies with "I don't know" score will be
        # 1.0 because the model is being faithful to its own response.
     "What were Apple's opening and closing stock prices on "
     "September 18, 2020?"

    )

    credentials = get_neo4j_credentials()

    result = retrieve(
        query=question,
        mode="graph_vector",
        credentials=credentials,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        model=MODEL,
    )

    actual_answer = result["answer"]

    # Basic failsafe: retrieval must actually return an answer.
    assert actual_answer, "Retrieval returned an empty answer."

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=result['retrieval_context'],
    )

    metric = FaithfulnessMetric(
        threshold=0.8,
        include_reason=True,
    )

    metric.measure(test_case)

    print("\n" + "=" * 60)
    print("ANSWER FAITHFULLNESS")
    print("=" * 60)
    print("Question:", question)
    print("Answer:", actual_answer)
    print("Score:", metric.score)
    print("Reason:", metric.reason)

    # Actual evaluation fails the test if relevancy is below threshold.
    assert metric.score is not None, "Answer relevancy produced no score."

    assert metric.score >= 0.8, (
        f"Answer relevancy failed. "
        f"Score={metric.score}, "
        f"Threshold=0.8, "
        f"Reason={metric.reason}"
    )

if __name__ == "__main__":
    test_answer_relevancy()