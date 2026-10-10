from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from dotenv import load_dotenv
from pathlib import Path
from deepeval import evaluate
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

def test_hallucinations():

    question = (
        # "What was the lowest recorded closing price of "
        # "Apple's stock during the pandemic, and on what date did it occur?"
        "Where is america?"
    )

    credentials = get_neo4j_credentials()

    result = retrieve(
        query=question,
        mode="graph_vector",
        credentials=credentials,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        model="openai_gpt_4_1_mini",
    )

    retrieval_context = result["retrieval_context"]
    actual_output = result["answer"]
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        context=retrieval_context,
    )

    metric = HallucinationMetric(
        threshold=0.8,
        include_reason=True,
    )

    evaluate(
        [test_case],
        [metric],
    )

    assert metric.score is not None, "Hallucination metric produced no score."
    assert metric.score >= 0.8, (
        f"Hallucination test failed.\n"
        f"Score: {metric.score}\n"
        f"Threshold: {metric.threshold}\n"
        f"Reason: {metric.reason}"
    )

if __name__ == "__main__":
    test_hallucinations()