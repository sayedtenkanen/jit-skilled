"""Optional learned-embeddings retrieval backend.

retrieval.py's TF-IDF retrieval is fine for a handful of items but doesn't
capture semantic similarity the way a real embedding model does (e.g. "net
income" vs. "profit" won't share any tokens). This module is a drop-in
alternative with the same `top_k_retrieve(target_question, pool, k,
exclude_task_id)` contract, backed by sentence-transformers.

It's an optional extra (`pip install jit-skilled[embeddings]`) rather than
a core dependency, to keep the default install dependency-light -- most of
this "lite" project doesn't need a ~100MB model download to run. Use
`--retrieval embeddings` on the CLI to opt in; `--retrieval tfidf` (the
default) needs nothing beyond PyYAML.

The ranking math (`rank_by_similarity`) is deliberately separated from the
model-loading/embedding code so it's unit-testable with plain Python lists
-- no sentence-transformers install required to test that the top-k
selection logic itself is correct.
"""
from __future__ import annotations

from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(target_vector: list[float], pool_vectors: list[list[float]],
                        k: int, exclude_idx: int | None = None) -> list[tuple[int, float]]:
    """Return up to k (index, score) pairs from pool_vectors, sorted by
    cosine similarity to target_vector, descending. Pure ranking logic --
    no embedding model involved, so this is testable without one.
    """
    scored = [
        (i, _cosine(target_vector, vec))
        for i, vec in enumerate(pool_vectors)
        if i != exclude_idx
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


class EmbeddingRetriever:
    """Loads a sentence-transformers model on first use and caches
    embeddings per unique question text (the demo pool is tiny, so an
    in-memory dict is enough -- no need for a real vector index).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "The 'embeddings' retrieval backend requires sentence-transformers, "
                "which is not installed. Install it with: "
                "pip install jit-skilled[embeddings]  (or use --retrieval tfidf, "
                "the zero-dependency default)."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, normalize_embeddings=True).tolist()
        return self._cache[text]

    def top_k_retrieve(self, target_question: str, pool: list[dict[str, Any]],
                        k: int = 3, exclude_task_id: str | None = None) -> list[dict[str, Any]]:
        candidates = [item for item in pool if item["task_id"] != exclude_task_id]
        if not candidates:
            return []
        pool_vectors = [self._embed(item["question"]) for item in candidates]
        target_vector = self._embed(target_question)
        ranked = rank_by_similarity(target_vector, pool_vectors, k)
        return [{**candidates[i], "score": score} for i, score in ranked]
