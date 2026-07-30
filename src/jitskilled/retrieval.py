"""Pure-Python TF-IDF cosine top-k retrieval over the evolve pool.

No embedding API required -- this is a stand-in for SkillTTA's embedding
retrieval, good enough for small pools. Swap in real embeddings if your
pool grows past a few hundred items.
"""
from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _build_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n = len(corpus_tokens)
    df: Counter = Counter()
    for tokens in corpus_tokens:
        df.update(set(tokens))
    return {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


def _vectorize(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (norm_a * norm_b)


def top_k_retrieve(target_question: str, pool: list[dict], k: int = 3,
                    exclude_task_id: str | None = None) -> list[dict]:
    """Return up to k pool items most similar to target_question, each with
    an added 'score' field, sorted descending. Excludes exclude_task_id.
    """
    candidates = [item for item in pool if item["task_id"] != exclude_task_id]
    if not candidates:
        return []

    corpus_tokens = [_tokenize(item["question"]) for item in candidates]
    idf = _build_idf(corpus_tokens + [_tokenize(target_question)])
    pool_vectors = [_vectorize(tokens, idf) for tokens in corpus_tokens]
    target_vector = _vectorize(_tokenize(target_question), idf)

    scored = [
        {**item, "score": _cosine(target_vector, vec)}
        for item, vec in zip(candidates, pool_vectors, strict=True)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
