from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return the L2-normalized version of ``vector``."""
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return vec / norm


def cosine_similarity(query: np.ndarray, candidates: Sequence[np.ndarray]) -> np.ndarray:
    """Compute cosine similarity between ``query`` and candidate vectors."""
    normalized_query = l2_normalize(query)
    normalized_candidates = np.stack([l2_normalize(vec) for vec in candidates])
    return normalized_candidates @ normalized_query


def top_k_cosine(
    query: np.ndarray,
    candidates: Sequence[np.ndarray],
    *,
    k: int,
) -> Iterable[tuple[int, float]]:
    """Yield the top-k cosine similarity scores."""
    if k <= 0:
        raise ValueError("k must be > 0")
    similarities = cosine_similarity(query, candidates)
    if len(similarities) == 0:
        return []
    top_indices = np.argpartition(-similarities, kth=min(k, len(similarities) - 1))[:k]
    sorted_indices = top_indices[np.argsort(-similarities[top_indices])]
    return [(int(idx), float(similarities[idx])) for idx in sorted_indices]