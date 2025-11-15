from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .hasher import LSHHasher
from .similarity import cosine_similarity, l2_normalize
from .storage import RedisStorage

VectorFetchFn = Callable[[Sequence[int]], np.ndarray]
IndexLike = Union[int, Sequence[int]]

__all__ = ["LSHRS"]


class LSHRS:
    """Redis-backed Locality Sensitive Hashing facade."""

    def __init__(
        self,
        *,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_prefix: str = "lsh",
        num_perm: int = 128,
        num_bands: Optional[int] = None,
        rows_per_band: Optional[int] = None,
        vector_dim: Optional[int] = None,
        vector_fetch_fn: Optional[VectorFetchFn] = None,
        buffer_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.storage = RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            prefix=redis_prefix,
        )
        self.redis = self.storage.client

        self.num_perm = num_perm
        self.num_bands, self.rows_per_band = self._resolve_band_params(
            num_perm=num_perm,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )

        self.vector_fetch = vector_fetch_fn
        self.buffer_size = buffer_size
        self.seed = seed

        self.vector_dim = vector_dim
        self._hasher: Optional[LSHHasher] = None
        if vector_dim is not None:
            self._init_hasher(vector_dim)

    def index(
        self,
        indices: Sequence[int],
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        """Batch index vectors, optionally fetching them via ``vector_fetch_fn``."""
        index_list = list(indices)
        if not index_list:
            return

        vectors_arr = self._resolve_vectors(index_list, vectors)
        self._ensure_hasher_ready(vectors_arr.shape[1])

        for start in range(0, len(index_list), self.buffer_size):
            end = min(start + self.buffer_size, len(index_list))
            batch_indices = index_list[start:end]
            batch_vectors = vectors_arr[start:end]
            self._index_batch(batch_indices, batch_vectors)

    def ingest(self, index: int, vector: np.ndarray) -> None:
        """Add a single vector to the index."""
        vec = self._prepare_vector(vector)
        self._ensure_hasher_ready(vec.shape[0])

        signatures = self._hash_vector(vec)
        for band_id, signature in enumerate(signatures):
            self.storage.add_to_bucket(band_id, signature, index)

    def query(
        self,
        vector: np.ndarray,
        *,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = None,
    ) -> List[Union[int, Tuple[int, float]]]:
        """Query for similar vectors."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be None or greater than zero")
        if top_p is not None and (top_p <= 0 or top_p > 100):
            raise ValueError("top_p must be in the (0, 100] interval")

        vec = self._prepare_vector(vector)
        self._ensure_hasher_ready(vec.shape[0])

        candidates = self._get_candidates(vec)
        if not candidates:
            return []

        if top_p is None:
            ordered = sorted(candidates)
            if top_k is None:
                return ordered
            return ordered[:top_k]

        if self.vector_fetch is None:
            raise ValueError("top_p queries require a vector_fetch_fn")
        return self._rerank_top_p(vec, candidates, top_p, top_k)

    def delete(self, indices: IndexLike) -> None:
        """Delete indices from all buckets."""
        if isinstance(indices, int):
            index_list = [indices]
        else:
            index_list = list(indices)
        if not index_list:
            return
        self.storage.remove_indices(index_list)

    # Internal helpers -----------------------------------------------------

    def _resolve_band_params(
        self,
        *,
        num_perm: int,
        num_bands: Optional[int],
        rows_per_band: Optional[int],
    ) -> Tuple[int, int]:
        if num_bands and rows_per_band:
            return num_bands, rows_per_band

        if num_bands:
            rows = max(1, num_perm // num_bands)
            return num_bands, rows

        if rows_per_band:
            bands = max(1, num_perm // rows_per_band)
            return bands, rows_per_band

        bands = max(1, int(math.sqrt(num_perm)))
        rows = max(1, num_perm // bands)
        if bands * rows < num_perm:
            rows += 1
        return bands, rows

    def _init_hasher(self, dim: int) -> None:
        self._hasher = LSHHasher(
            num_bands=self.num_bands,
            rows_per_band=self.rows_per_band,
            dim=dim,
            seed=self.seed,
        )

    def _ensure_hasher_ready(self, dim: int) -> None:
        if self.vector_dim is None:
            self.vector_dim = dim
            self._init_hasher(dim)
        elif self.vector_dim != dim:
            raise ValueError(
                f"Vector dimensionality mismatch: expected {self.vector_dim}, got {dim}"
            )
        elif self._hasher is None:
            self._init_hasher(dim)

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        return arr

    def _resolve_vectors(
        self,
        indices: Sequence[int],
        vectors: Optional[np.ndarray],
    ) -> np.ndarray:
        if vectors is None:
            if self.vector_fetch is None:
                raise ValueError("Vectors or vector_fetch_fn required")
            fetched = self.vector_fetch(indices)
            arr = np.asarray(fetched, dtype=np.float32)
        else:
            arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError("Input vectors must be a 2D array")
        if arr.shape[0] != len(indices):
            raise ValueError(
                f"Expected {len(indices)} vectors, received {arr.shape[0]}"
            )
        return arr

    def _hash_vector(self, vector: np.ndarray):
        if self._hasher is None:
            raise RuntimeError("Hasher not initialized")
        return self._hasher.hash_vector(vector)

    def _index_batch(self, indices: Sequence[int], vectors: np.ndarray) -> None:
        operations: List[Tuple[int, bytes, int]] = []
        for idx, vector in zip(indices, vectors, strict=True):
            signatures = self._hash_vector(vector)
            for band_id, signature in enumerate(signatures):
                operations.append((band_id, signature, idx))
        self.storage.batch_add(operations)

    def _get_candidates(self, vector: np.ndarray) -> Set[int]:
        signatures = self._hash_vector(vector)
        candidates: Set[int] = set()
        for band_id, signature in enumerate(signatures):
            bucket = self.storage.get_bucket(band_id, signature)
            candidates.update(bucket)
        return candidates

    def _rerank_top_p(
        self,
        query: np.ndarray,
        candidates: Set[int],
        top_p: float,
        top_k: Optional[int],
    ) -> List[Tuple[int, float]]:
        candidate_list = sorted(candidates)
        candidate_vectors = self.vector_fetch(candidate_list)
        candidate_arr = np.asarray(candidate_vectors, dtype=np.float32)

        if candidate_arr.ndim != 2 or candidate_arr.shape[1] != query.shape[0]:
            raise ValueError("Fetched vectors do not match query dimensionality")

        normalized_query = l2_normalize(query)
        normalized_candidates = np.apply_along_axis(l2_normalize, 1, candidate_arr)
        similarities = normalized_candidates @ normalized_query

        percentile_threshold = np.percentile(similarities, 100 - top_p)
        selected = [
            (candidate_list[i], float(similarities[i]))
            for i in range(len(candidate_list))
            if similarities[i] >= percentile_threshold
        ]
        selected.sort(key=lambda item: item[1], reverse=True)

        if top_k is not None:
            return selected[:top_k]
        return selected

    # Convenience ----------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries under the configured Redis prefix."""
        self.storage.clear()

    def stats(self) -> dict:
        """Return lightweight stats for observability."""
        return {
            "num_bands": self.num_bands,
            "rows_per_band": self.rows_per_band,
            "vector_dim": self.vector_dim,
            "buffer_size": self.buffer_size,
        }