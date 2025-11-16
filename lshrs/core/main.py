from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from lshrs.hash.lsh import LSHHasher
from lshrs.storage.redis import BucketOperation, RedisStorage
from lshrs.utils.br import get_optimal_config
from lshrs.utils.similarity import top_k_cosine


VectorFetchFn = Callable[[Sequence[int]], np.ndarray]
CandidateScores = List[Tuple[int, float]]
Loader = Callable[..., Iterable[Tuple[Sequence[int], np.ndarray]]]


class LSHRS:
    """
    High-level orchestrator for the Redis-backed Locality Sensitive Hashing pipeline.

    The class coordinates three core responsibilities:

    1. Hash vector representations using configurable random projections.
    2. Persist hash buckets inside Redis for fast candidate retrieval.
    3. Provide query helpers that optionally rerank candidates by cosine similarity.

    Parameters
    ----------
    dim:
        Dimensionality of the vectors being indexed. Must remain constant for the
        lifetime of the instance.
    num_perm:
        Total number of random projections (hash bits). Defaults to 128.
    num_bands / rows_per_band:
        Optional explicit banding configuration. When omitted the module selects
        optimal values via :func:`lshrs.utils.br.get_optimal_config`.
    similarity_threshold:
        Target similarity used during automatic band/row selection.
    buffer_size:
        Number of Redis bucket operations to accumulate before pipelining them.
    vector_fetch_fn:
        Callable that retrieves dense vectors from the primary data store. Required
        for reranking (``top_p`` queries) and optional for pure hash lookups.
    storage:
        Custom :class:`lshrs.storage.redis.RedisStorage` instance. When omitted a new
        connection is created based on the provided Redis keyword arguments.
    redis_host / redis_port / redis_db / redis_password / redis_prefix / decode_responses:
        Redis connection configuration passed to :class:`RedisStorage` when ``storage``
        is not supplied.
    seed:
        Random seed used to generate the projection matrices.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_perm: int = 128,
        num_bands: Optional[int] = None,
        rows_per_band: Optional[int] = None,
        similarity_threshold: float = 0.5,
        buffer_size: int = 10_000,
        vector_fetch_fn: Optional[VectorFetchFn] = None,
        storage: Optional[RedisStorage] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_prefix: str = "lsh",
        decode_responses: bool = False,
        seed: int = 42,
    ) -> None:
        if dim <= 0:
            raise ValueError("Vector dimensionality must be greater than zero")
        if num_perm <= 0:
            raise ValueError("num_perm must be greater than zero")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than zero")

        auto_config = num_bands is None or rows_per_band is None
        if auto_config:
            num_bands, rows_per_band = get_optimal_config(num_perm, similarity_threshold)
        assert num_bands is not None and rows_per_band is not None
        if num_bands * rows_per_band != num_perm:
            raise ValueError(
                "num_bands * rows_per_band must equal num_perm "
                f"(received {num_bands} * {rows_per_band} != {num_perm})"
            )

        self._dim = dim
        self._buffer_size = buffer_size
        self._vector_fetch_fn = vector_fetch_fn
        self._hasher = LSHHasher(
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            dim=dim,
            seed=seed,
        )
        self._storage = storage or RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=decode_responses,
            prefix=redis_prefix,
        )
        self._buffer: List[BucketOperation] = []

        self._config: Dict[str, Any] = {
            "dim": dim,
            "num_perm": num_perm,
            "num_bands": num_bands,
            "rows_per_band": rows_per_band,
            "similarity_threshold": similarity_threshold,
            "buffer_size": buffer_size,
            "seed": seed,
        }
        self._redis_config: Dict[str, Any] = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "password": redis_password,
            "prefix": redis_prefix,
            "decode_responses": decode_responses,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            "LSHRS("
            f"dim={self._dim}, "
            f"num_perm={self._config['num_perm']}, "
            f"num_bands={self._config['num_bands']}, "
            f"rows_per_band={self._config['rows_per_band']}, "
            f"redis_prefix='{self._redis_config['prefix']}'"
            ")"
        )

    # ---------------------------------------------------------------------
    # Public ingestion API
    # ---------------------------------------------------------------------

    def create_signatures(self, *, format: str = "postgres", **loader_kwargs: Any) -> None:
        """
        Bulk-ingest vectors using one of the built-in IO helpers.

        Parameters
        ----------
        format:
            Data source identifier. Supported values are ``"postgres"`` and ``"parquet"``.
        **loader_kwargs:
            Keyword arguments forwarded to the selected loader implementation. See
            :mod:`lshrs.io.postgres` and :mod:`lshrs.io.parquet` for details.
        """
        loader = self._resolve_loader(format)
        for indices, vectors in loader(**loader_kwargs):
            self.index(indices, vectors)

    def ingest(self, index: int, vector: np.ndarray) -> None:
        """
        Insert a single vector into the index.

        Parameters
        ----------
        index:
            Integer identifier used by the backing datastore.
        vector:
            Dense numpy array representation of the item to index.
        """
        if index < 0:
            raise ValueError("index must be non-negative")

        vector_arr = self._prepare_vector(vector)
        signatures = self._hasher.hash_vector(vector_arr)
        self._enqueue_operations(index, signatures)
        self._flush_buffer_if_needed()

    def index(self, indices: Sequence[int], vectors: Optional[np.ndarray] = None) -> None:
        """
        Batch-ingest vectors by hashing and storing them in Redis buckets.

        Parameters
        ----------
        indices:
            Sequence of integer identifiers. Order must align with ``vectors``.
        vectors:
            Optional 2D numpy array of shape ``(len(indices), dim)``. When omitted the
            instance-wide ``vector_fetch_fn`` is used to retrieve the vectors.
        """
        if not indices:
            return

        if vectors is None:
            fetch_fn = self._require_vector_fetch_fn()
            vectors = fetch_fn(indices)

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(
                f"Vectors must have shape (n, {self._dim}); received {arr.shape}"
            )
        if arr.shape[0] != len(indices):
            raise ValueError(
                "Number of vectors does not match number of indices "
                f"(received {arr.shape[0]} vectors for {len(indices)} indices)"
            )

        for idx, vec in zip(indices, arr):
            self.ingest(int(idx), vec)

        self._flush_buffer()

    # ---------------------------------------------------------------------
    # Query helpers
    # ---------------------------------------------------------------------

    def query(
        self,
        vector: np.ndarray,
        *,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = None,
    ) -> Union[List[int], CandidateScores]:
        """
        Retrieve candidates similar to ``vector``.

        Parameters
        ----------
        vector:
            Query embedding.
        top_k:
            Maximum number of candidates to return. When ``top_p`` is also supplied, the
            result count is the minimum of both constraints. ``None`` returns all results.
        top_p:
            When provided, rerank candidates by cosine similarity and return the smallest
            prefix whose cumulative proportion of results covers ``top_p`` of the pool.
            Requires ``vector_fetch_fn`` to be configured.

        Returns
        -------
        list[int] | list[tuple[int, float]]
            Pure top-k mode returns candidate indices ordered by band collisions.
            Top-p mode returns (index, score) pairs ordered by cosine similarity.
        """
        query_vector = self._prepare_vector(vector)
        candidate_counts = self._candidate_counts(query_vector)
        if not candidate_counts:
            return []

        ordered = sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))

        if top_p is None:
            if top_k is None:
                top_k = len(ordered)
            if top_k <= 0:
                raise ValueError("top_k must be greater than zero when provided")
            return [idx for idx, _ in ordered[:top_k]]

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be within the range (0, 1]")

        candidate_indices = [idx for idx, _ in ordered]
        fetch_fn = self._require_vector_fetch_fn()
        candidate_vectors = fetch_fn(candidate_indices)
        arr = np.asarray(candidate_vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(
                f"Fetched vectors must have shape (n, {self._dim}); received {arr.shape}"
            )
        if arr.shape[0] != len(candidate_indices):
            raise ValueError(
                "vector_fetch_fn returned mismatched batch size "
                f"(expected {len(candidate_indices)}, received {arr.shape[0]})"
            )

        similarities = top_k_cosine(query_vector, arr, k=len(candidate_indices))
        ordered_scores = [(candidate_indices[pos], score) for pos, score in similarities]

        limit = max(1, math.ceil(len(ordered_scores) * top_p))
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be greater than zero when provided")
            limit = min(limit, top_k)

        return ordered_scores[:limit]

    def get_top_k(self, vector: np.ndarray, topk: int = 10) -> List[int]:
        """
        Convenience wrapper around :meth:`query` for pure top-k retrieval.
        """
        results = self.query(vector, top_k=topk, top_p=None)
        return list(results)  # type: ignore[return-value]

    def get_above_p(self, vector: np.ndarray, p: float = 0.95) -> CandidateScores:
        """
        Convenience wrapper around :meth:`query` for top-p reranking.
        """
        results = self.query(vector, top_k=None, top_p=p)
        return list(results)  # type: ignore[return-value]

    # ---------------------------------------------------------------------
    # Maintenance helpers
    # ---------------------------------------------------------------------

    def delete(self, indices: Union[int, Sequence[int]]) -> None:
        """
        Remove one or more indices from every Redis bucket.
        """
        if isinstance(indices, int):
            to_remove = [indices]
        else:
            to_remove = [int(idx) for idx in indices]
        self._storage.remove_indices(to_remove)

    def clear(self) -> None:
        """
        Delete every key associated with the configured Redis prefix.
        """
        self._flush_buffer()
        self._storage.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Return the current configuration snapshot.
        """
        return {
            "dimension": self._dim,
            "num_perm": self._config["num_perm"],
            "num_bands": self._config["num_bands"],
            "rows_per_band": self._config["rows_per_band"],
            "buffer_size": self._buffer_size,
            "similarity_threshold": self._config["similarity_threshold"],
            "redis_prefix": self._redis_config["prefix"],
        }

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

    def save_to_disk(self, path: Union[str, Path]) -> None:
        """
        Persist configuration and projection matrices to disk.

        The serialized payload intentionally excludes the Redis connection so that the
        restored instance can target a different Redis deployment.
        """
        self._flush_buffer()
        payload = self.__getstate__()
        Path(path).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load_from_disk(
        cls,
        path: Union[str, Path],
        *,
        redis_config: Optional[Dict[str, Any]] = None,
        vector_fetch_fn: Optional[VectorFetchFn] = None,
        storage: Optional[RedisStorage] = None,
    ) -> "LSHRS":
        """
        Restore an :class:`LSHRS` instance previously saved via :meth:`save_to_disk`.

        Parameters
        ----------
        path:
            Path to the serialized payload.
        redis_config:
            Optional overrides for the stored Redis configuration.
        vector_fetch_fn:
            Callable to attach to the restored instance.
        storage:
            Optional pre-initialised :class:`RedisStorage`. When given the provided
            ``redis_config`` overrides are still persisted for introspection purposes.
        """
        state = pickle.loads(Path(path).read_bytes())
        config = state["config"]
        stored_redis = state["redis_config"].copy()
        if redis_config:
            stored_redis.update(redis_config)

        instance = cls(
            dim=config["dim"],
            num_perm=config["num_perm"],
            num_bands=config["num_bands"],
            rows_per_band=config["rows_per_band"],
            similarity_threshold=config["similarity_threshold"],
            buffer_size=config["buffer_size"],
            vector_fetch_fn=vector_fetch_fn,
            storage=storage,
            redis_host=stored_redis["host"],
            redis_port=stored_redis["port"],
            redis_db=stored_redis["db"],
            redis_password=stored_redis["password"],
            redis_prefix=stored_redis["prefix"],
            decode_responses=stored_redis["decode_responses"],
            seed=config["seed"],
        )
        instance._hasher.projections = [
            np.asarray(matrix, dtype=np.float32) for matrix in state["projections"]
        ]
        return instance

    # ---------------------------------------------------------------------
    # Pickle protocol
    # ---------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """
        Return a minimal, pickle-friendly representation of the instance.
        """
        self._flush_buffer()
        return {
            "config": self._config.copy(),
            "redis_config": self._redis_config.copy(),
            "projections": [
                np.asarray(matrix, dtype=np.float32) for matrix in self._hasher.projections
            ],
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        restored = self.__class__(
            dim=state["config"]["dim"],
            num_perm=state["config"]["num_perm"],
            num_bands=state["config"]["num_bands"],
            rows_per_band=state["config"]["rows_per_band"],
            similarity_threshold=state["config"]["similarity_threshold"],
            buffer_size=state["config"]["buffer_size"],
            vector_fetch_fn=None,
            redis_host=state["redis_config"]["host"],
            redis_port=state["redis_config"]["port"],
            redis_db=state["redis_config"]["db"],
            redis_password=state["redis_config"]["password"],
            redis_prefix=state["redis_config"]["prefix"],
            decode_responses=state["redis_config"]["decode_responses"],
            seed=state["config"]["seed"],
        )
        self.__dict__ = restored.__dict__
        self._hasher.projections = [
            np.asarray(matrix, dtype=np.float32) for matrix in state["projections"]
        ]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self._dim:
            raise ValueError(
                f"Vector must have dimension {self._dim}; received {arr.shape[0]}"
            )
        return arr

    def _candidate_counts(self, query_vector: np.ndarray) -> Dict[int, int]:
        signatures = self._hasher.hash_vector(query_vector)
        counts: Dict[int, int] = {}
        for band_id, hash_val in enumerate(signatures):
            for candidate in self._storage.get_bucket(band_id, hash_val):
                counts[candidate] = counts.get(candidate, 0) + 1
        return counts

    def _enqueue_operations(
        self,
        index: int,
        signatures: Iterable[bytes],
    ) -> None:
        for band_id, hash_val in enumerate(signatures):
            self._buffer.append((band_id, hash_val, int(index)))
        # Buffer is flushed lazily to leverage pipelining.

    def _flush_buffer_if_needed(self) -> None:
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        self._storage.batch_add(self._buffer)
        self._buffer.clear()

    def _require_vector_fetch_fn(self) -> VectorFetchFn:
        if self._vector_fetch_fn is None:
            raise RuntimeError(
                "vector_fetch_fn must be supplied for operations requiring reranking"
            )
        return self._vector_fetch_fn

    def _resolve_loader(self, format: str) -> Loader:
        normalized = format.lower()
        if normalized in {"postgres", "pg"}:
            from lshrs.io.postgres import iter_postgres_vectors

            return iter_postgres_vectors
        if normalized in {"parquet", "pq"}:
            from lshrs.io.parquet import iter_parquet_vectors

            return iter_parquet_vectors
        raise ValueError(f"Unsupported signature creation format '{format}'")


# Backwards compatibility alias for earlier versions that used lowercase naming.
lshrs = LSHRS