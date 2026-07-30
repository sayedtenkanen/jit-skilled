"""Tests for retrieval_embeddings.py.

rank_by_similarity is pure Python arithmetic over plain lists, so it's
tested directly with synthetic vectors -- no sentence-transformers install
required. EmbeddingRetriever itself (which needs the real, optional
dependency) is only tested for its graceful-failure path: sentence-
transformers is intentionally not part of the default install, and this
sandbox doesn't have it, so the ImportError path below is exercised for
real rather than mocked.
"""
import pytest

from jitskilled.retrieval_embeddings import EmbeddingRetriever, rank_by_similarity


def test_rank_by_similarity_orders_by_cosine_descending():
    target = [1.0, 0.0]
    pool = [
        [0.0, 1.0],   # orthogonal -> similarity 0
        [1.0, 0.0],   # identical -> similarity 1
        [0.7, 0.7],   # 45 degrees -> similarity ~0.707
    ]
    ranked = rank_by_similarity(target, pool, k=3)
    order = [i for i, _score in ranked]
    assert order == [1, 2, 0]


def test_rank_by_similarity_respects_k():
    target = [1.0, 0.0]
    pool = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    ranked = rank_by_similarity(target, pool, k=2)
    assert len(ranked) == 2


def test_rank_by_similarity_excludes_index():
    target = [1.0, 0.0]
    pool = [[1.0, 0.0], [0.0, 1.0]]
    ranked = rank_by_similarity(target, pool, k=2, exclude_idx=0)
    assert [i for i, _ in ranked] == [1]


def test_rank_by_similarity_handles_zero_vector():
    # A zero vector has undefined cosine similarity -- must not divide by
    # zero or crash, just score it as unrelated (0.0).
    target = [1.0, 0.0]
    pool = [[0.0, 0.0], [1.0, 0.0]]
    ranked = rank_by_similarity(target, pool, k=2)
    scores = dict(ranked)
    assert scores[0] == 0.0
    assert scores[1] == 1.0


def test_embedding_retriever_raises_clear_error_without_dependency():
    # sentence-transformers is an optional extra; this test only makes sense
    # when it's NOT installed. If it IS installed, skip.
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed; "
                    "missing-dependency path not exercisable")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="jit-skilled\\[embeddings\\]"):
        EmbeddingRetriever()
