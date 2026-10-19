"""Tests of the local embedder (fastembed + BAAI/bge-small-en-v1.5). Needs the model on disk: the first run
downloads about 70 MB, later runs are offline. No API key and no database."""
import math

import pytest

from pipeline.graphdb.db import LocalEmbeddingFunction, load_embedding_model
from pipeline.shared.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER


@pytest.fixture(scope="module")
def embedder():
    embeddings, dimension = load_embedding_model(DEFAULT_EMBEDDING_PROVIDER, DEFAULT_EMBEDDING_MODEL)
    return embeddings, dimension


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_defaults_are_the_local_bge_small_model():
    assert DEFAULT_EMBEDDING_PROVIDER == "local"
    assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_load_returns_a_384_dimension_embedder(embedder):
    embeddings, dimension = embedder
    assert isinstance(embeddings, LocalEmbeddingFunction)
    assert dimension == 384
    vector = embeddings.embed_query("def charge(card): return gateway.post(card)")
    assert len(vector) == dimension
    assert all(isinstance(v, float) for v in vector)


def test_provider_name_is_case_insensitive():
    _, dimension = load_embedding_model("LOCAL", DEFAULT_EMBEDDING_MODEL)
    assert dimension == 384


@pytest.mark.parametrize("provider,model", [
    ("openai", "text-embedding-3-small"),     # the old provider must not silently come back
    ("local", "text-embedding-3-small"),
    ("local", "not-a-model"),
    ("nope", DEFAULT_EMBEDDING_MODEL),
])
def test_unsupported_provider_or_model_is_rejected(provider, model):
    with pytest.raises(ValueError, match="Unsupported provider/model"):
        load_embedding_model(provider, model)


def test_embedding_is_deterministic(embedder):
    embeddings, _ = embedder
    assert embeddings.embed_query("unsafe deserialization") == pytest.approx(embeddings.embed_query("unsafe deserialization"))


def test_newlines_are_treated_as_spaces(embedder):
    embeddings, _ = embedder
    assert embeddings.embed_query("line one\nline two") == pytest.approx(embeddings.embed_query("line one line two"))


@pytest.mark.parametrize("text", ["", None])
def test_empty_text_still_gets_a_vector(embedder, text):
    embeddings, dimension = embedder
    assert len(embeddings.embed_query(text)) == dimension


def test_documents_batch_matches_single_queries(embedder):
    embeddings, dimension = embedder
    texts = ["SQL built from user input", "mutable default argument", "hardcoded password"]
    batch = embeddings.embed_documents(texts)
    assert len(batch) == 3 and all(len(v) == dimension for v in batch)
    for text, vector in zip(texts, batch):
        assert cosine(vector, embeddings.embed_query(text)) > 0.999


def test_the_model_is_loaded_once_and_shared():
    first = LocalEmbeddingFunction(DEFAULT_EMBEDDING_MODEL)
    second = LocalEmbeddingFunction(DEFAULT_EMBEDDING_MODEL)
    assert first._embedder is second._embedder


def test_related_text_ranks_above_unrelated_text(embedder):
    embeddings, _ = embedder
    query = embeddings.embed_query("unsafe database query built from user input")
    related = embeddings.embed_query("SQL injection via string concatenation in a query")
    unrelated = embeddings.embed_query("How do I bake sourdough bread?")
    assert cosine(query, related) > cosine(query, unrelated) + 0.2
