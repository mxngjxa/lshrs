"""
High-Level LSH Pipeline Orchestrator with Redis Backend

This module provides the main LSHRS class that ties together all LSH components:
- Vector hashing using random projections (LSHHasher)
- Bucket storage in Redis (RedisStorage)
- Similarity search with optional reranking
- Bulk ingestion from various data sources
- Index persistence and restoration

The LSHRS class is the primary interface for building and querying LSH indices.
It handles the complete workflow from vector ingestion to candidate retrieval
and similarity-based reranking.

Architecture overview:
    1. Vectors → LSHHasher → Hash signatures (one per band)
    2. Signatures → RedisStorage → Buckets (Redis sets)
    3. Query → Get candidates from buckets → Optional reranking → Results

Key features:
    - Automatic band/row configuration for target similarity
    - Buffered ingestion for high throughput
    - Top-k and top-p query modes
    - Index persistence (save/load projections)
    - Multiple data source loaders (PostgreSQL, Parquet)
"""

from __future__ import annotations

import json
import logging
import math
# import pickle
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from lshrs.hash.lsh import LSHHasher
from lshrs.storage.redis import BucketOperation, RedisStorage
from lshrs.utils.br import get_optimal_config
from lshrs.utils.similarity import top_k_cosine

logger = logging.getLogger(__name__)

# Type aliases for clarity and documentation
# Function that fetches vectors given their indices (for reranking)
VectorFetchFn = Callable[[Sequence[int]], np.ndarray]

# List of (index, similarity_score) tuples returned by similarity queries
CandidateScores = List[Tuple[int, float]]

# Generic loader function that yields (indices, vectors) batches
Loader = Callable[..., Iterable[Tuple[Sequence[int], np.ndarray]]]


class LSHRS:
    """
    High-level orchestrator for the Redis-backed Locality Sensitive Hashing pipeline.

    The LSHRS class provides a complete LSH implementation with three core responsibilities:

    1. **Hash Generation**: Convert vectors to binary signatures using random projections
    2. **Storage Management**: Persist hash buckets in Redis for fast retrieval
    3. **Query Processing**: Find similar vectors with optional cosine similarity reranking

    The class supports both exact top-k retrieval (return k most similar) and
    probabilistic top-p retrieval (return candidates covering p% of similarity mass).

    Parameters
    ----------
    dim : int
        Dimensionality of vectors being indexed. Must remain constant for the
        lifetime of the instance. All vectors must have exactly this dimension.

    num_perm : int, default=128
        Total number of random projections (hash bits). More bits = better accuracy
        but slower hashing and more storage. Common values: 64, 128, 256.

    num_bands : int, optional
        Number of independent hash bands (hash tables). More bands = higher recall.
        If not specified, automatically computed from num_perm and similarity_threshold.
        Must satisfy: num_bands × rows_per_band = num_perm.

    rows_per_band : int, optional
        Number of hash bits per band. More rows = higher precision (fewer false positives).
        If not specified, automatically computed from num_perm and similarity_threshold.

    similarity_threshold : float, default=0.5
        Target Jaccard similarity for automatic band/row selection. Used only when
        num_bands or rows_per_band is not explicitly provided. Range: (0, 1).
        Lower values = more aggressive matching, higher recall.

    buffer_size : int, default=8912
        Number of Redis operations to accumulate before pipelining. Larger buffers
        improve throughput but increase memory usage and latency. Sweet spot: 1k-100k.

    vector_fetch_fn : callable, optional
        Function that retrieves vectors from primary storage given their indices.
        Signature: fn(indices: Sequence[int]) -> np.ndarray of shape (len(indices), dim).
        Required for reranking queries (top_p mode). Can be database query, file read, etc.

    storage : RedisStorage, optional
        Pre-configured Redis storage instance. If provided, redis_* parameters are ignored.
        Useful for sharing Redis connections or custom configurations.

    redis_host : str, default="localhost"
        Redis server hostname or IP address. Use "localhost" for local development.

    redis_port : int, default=6379
        Redis server port. Standard Redis port is 6379.

    redis_db : int, default=0
        Redis database number (0-15). Use different DBs for isolation.

    redis_password : str, optional
        Redis authentication password. None if no AUTH required.

    redis_prefix : str, default="lsh"
        Key prefix for all Redis operations. Use unique prefixes for different indices.
        Must not contain colons (:) to avoid key parsing issues.

    decode_responses : bool, default=False
        Whether Redis should decode responses to strings. Keep False for binary data.

    seed : int, default=42
        Random seed for projection matrix generation. Use same seed for reproducibility.

    Examples
    --------
    Basic usage with automatic configuration:

    >>> lsh = LSHRS(dim=128, similarity_threshold=0.7)
    >>> lsh.ingest(0, np.random.randn(128))
    >>> candidates = lsh.get_top_k(query_vector, topk=10)

    Custom configuration with reranking:

    >>> def fetch_vectors(indices):
    ...     # Load from database, file, etc.
    ...     return vectors_array
    >>>
    >>> lsh = LSHRS(
    ...     dim=768,
    ...     num_bands=20,
    ...     rows_per_band=10,
    ...     vector_fetch_fn=fetch_vectors
    ... )
    >>> # Get top 10% most similar with cosine reranking
    >>> results = lsh.get_above_p(query_vector, p=0.1)
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
        redis_max_connections: int = 50,
        decode_responses: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Create and configure an LSH orchestrator that manages hashing, Redis storage, and buffered ingestion.
        
        Initializes internal state including the LSHHasher (with random projections), a RedisStorage instance (or uses the provided one), an in-memory buffer and lock for batched bucket operations, and configuration dictionaries used for persistence and introspection.
        
        Parameters:
            dim: Dimensionality of vectors; must be greater than zero.
            num_perm: Total number of projection bits used by the hasher.
            num_bands, rows_per_band: If both provided, must satisfy num_bands * rows_per_band == num_perm. If either is omitted, they are auto-configured from num_perm and similarity_threshold.
            similarity_threshold: Target similarity used to auto-configure bands/rows when needed.
            buffer_size: Number of buffered bucket operations that triggers an automatic flush; must be greater than zero.
            vector_fetch_fn: Optional callable to fetch vectors by indices; required for reranking/top-p query modes.
            storage: Optional preconfigured RedisStorage instance; if omitted, a RedisStorage is created using the redis_* parameters.
            redis_host, redis_port, redis_db, redis_password, redis_prefix, decode_responses: Redis connection and key-prefix configuration used when creating a RedisStorage.
            redis_max_connections: Maximum number of connections allowed in the Redis connection pool when creating RedisStorage.
            seed: Random seed used to initialize the hasher's projection matrices.
        """
        # Validate basic parameters
        if dim <= 0:
            raise ValueError("Vector dimensionality must be greater than zero")
        if num_perm <= 0:
            raise ValueError("num_perm must be greater than zero")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than zero")

        # Auto-configure bands/rows if not explicitly provided
        # Uses probability theory to find optimal configuration for target similarity
        auto_config = num_bands is None or rows_per_band is None
        if auto_config:
            num_bands, rows_per_band = get_optimal_config(
                num_perm, similarity_threshold
            )

        # Type narrowing for mypy - guaranteed non-None after this
        if num_bands is None or rows_per_band is None:
            raise RuntimeError(
                f"Auto-config failed: get_optimal_config({num_perm}, {similarity_threshold}) "
                f"-> ({num_bands}, {rows_per_band})"
            )

        # Validate band/row configuration matches total hash bits
        if num_bands * rows_per_band != num_perm:
            raise ValueError(
                "num_bands * rows_per_band must equal num_perm "
                f"(received {num_bands} * {rows_per_band} != {num_perm})"
            )

        # Store core parameters
        self._dim = dim
        self._buffer_size = buffer_size
        self._vector_fetch_fn = vector_fetch_fn

        # Initialize the LSH hasher with random projections
        self._hasher = LSHHasher(
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            dim=dim,
            seed=seed,
        )

        # Set up Redis storage (use provided instance or create new one)
        self._storage = storage or RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=decode_responses,
            prefix=redis_prefix,
            max_connections=redis_max_connections,
        )

        # Initialize operation buffer for batch processing
        self._buffer: List[BucketOperation] = []
        self._buffer_lock = Lock()

        # Store configuration for persistence and introspection
        self._config: Dict[str, Any] = {
            "dim": dim,
            "num_perm": num_perm,
            "num_bands": num_bands,
            "rows_per_band": rows_per_band,
            "similarity_threshold": similarity_threshold,
            "buffer_size": buffer_size,
            "seed": seed,
        }

        # Store Redis config separately for potential override during load
        self._redis_config: Dict[str, Any] = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "password": redis_password,
            "prefix": redis_prefix,
            "decode_responses": decode_responses,
            "max_connections": redis_max_connections,
        }

    def close(self) -> None:
        """
        Flush any buffered bucket operations and close the Redis storage connection.
        
        Ensures pending operations are executed before closing the underlying Redis client to avoid leaving unflushed data or open connections.
        """
        self.flush()
        self._storage.close()

    def __enter__(self) -> "LSHRS":
        """
        Enter runtime context for the LSHRS instance.
        
        Returns:
            self (LSHRS): The LSHRS instance to be used within the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context manager and close internal resources.
        
        This method is called on context exit and ensures buffered operations are flushed and the underlying Redis connection is closed.
        """
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - convenience
        """
        Concise string describing the instance's key LSHRS configuration.
        
        Returns:
            str: Representation in the form "LSHRS(dim=..., num_perm=..., num_bands=..., rows_per_band=..., redis_prefix='...')".
        """
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

    def create_signatures(
        self, *, format: str = "postgres", **loader_kwargs: Any
    ) -> None:
        """
        Bulk-ingest vectors using one of the built-in IO helpers.

        This method provides a high-level interface for loading vectors from
        common data sources. The actual loading logic is delegated to format-specific
        loaders that yield (indices, vectors) batches.

        Supported formats:
            - "postgres" / "pg": Load from PostgreSQL database
            - "parquet" / "pq": Load from Parquet files

        The loaders handle:
            - Efficient batch reading
            - Memory management for large datasets
            - Proper type conversion to numpy arrays

        Parameters
        ----------
        format : str, default="postgres"
            Data source identifier. Case-insensitive.
            Options: "postgres", "pg", "parquet", "pq"

        **loader_kwargs : Any
            Format-specific parameters passed to the loader.

            For PostgreSQL (see lshrs.io.postgres):
                - connection_string: Database URL
                - query: SQL query to fetch vectors
                - batch_size: Number of rows per fetch

            For Parquet (see lshrs.io.parquet):
                - path: File or directory path
                - columns: Vector column names
                - batch_size: Rows per batch

        Examples
        --------
        Load from PostgreSQL:

        >>> lsh.create_signatures(
        ...     format="postgres",
        ...     connection_string="postgresql://localhost/mydb",
        ...     query="SELECT id, embedding FROM vectors",
        ...     batch_size=1000
        ... )

        Load from Parquet:

        >>> lsh.create_signatures(
        ...     format="parquet",
        ...     path="/data/embeddings.parquet",
        ...     columns=["vector"],
        ...     batch_size=5000
        ... )

        Raises
        ------
        ValueError
            If format is not supported.
        ImportError
            If required dependencies for format are not installed.
        """
        # Resolve format string to loader function
        loader = self._resolve_loader(format)

        # Process vectors in batches
        # Loader yields (indices, vectors) tuples
        for indices, vectors in loader(**loader_kwargs):
            self.index(indices, vectors)

    def ingest(self, index: int, vector: np.ndarray) -> None:
        """
        Insert a single vector into the LSH index under the given integer identifier.
        
        Parameters:
            index (int): Non-negative integer identifier for the vector.
            vector (np.ndarray): Dense vector convertible to a 1-D float32 numpy array of length equal to the LSH dimensionality.
        
        Raises:
            ValueError: If `index` is negative, if the vector's dimensionality does not equal the configured dimension, or if the vector is all zeros.
        """
        # Validate index is non-negative
        if index < 0:
            raise ValueError("index must be non-negative")

        # Prepare vector (validate dimension, convert to float32)
        vector_arr = self._prepare_vector(vector)

        # Generate hash signatures for all bands
        signatures = self._hasher.hash_vector(vector_arr)

        # Add operations to buffer (one per band)
        self._enqueue_operations(index, signatures)

        # Flush if buffer is full
        self._flush_buffer_if_needed()

    def flush(self) -> None:
        """
        Execute all buffered operations via Redis pipeline.

        Sends all accumulated bucket operations in a single batch for
        efficiency. Clears buffer after execution.

        This method should be called manually if you need to ensure all ingested
        vectors are immediately searchable, although it is called automatically
        when the buffer fills up or when the LSHRS instance is closed.
        """
        with self._buffer_lock:
            if not self._buffer:
                return
            # Copy buffer content to release lock quickly
            ops_to_flush = list(self._buffer)
            self._buffer.clear()

        # Batch execute all operations outside the lock
        # This prevents blocking ingestion while waiting for Redis
        try:
            self._storage.batch_add(ops_to_flush)
        except Exception as e:
            logger.error(f"Failed to flush buffer to Redis: {e}")
            with self._buffer_lock:
                # Restore operations to the front of the buffer
                self._buffer[0:0] = ops_to_flush
            raise

    def index(
        self, indices: Sequence[int], vectors: Optional[np.ndarray] = None
    ) -> None:
        """
        Batch-ingest vectors by hashing and storing them in Redis buckets.

        This is the recommended method for indexing multiple vectors. It's more
        efficient than calling ingest() in a loop because it:
            - Validates all vectors upfront
            - Uses buffered Redis operations
            - Ensures atomic batch completion with final flush

        Two modes of operation:
            1. Direct: Provide vectors as numpy array
            2. Fetch: Use vector_fetch_fn to load vectors given indices

        Parameters
        ----------
        indices : Sequence[int]
            Integer identifiers for vectors. Order must match vectors array.
            All indices should be unique to avoid overwriting.

        vectors : np.ndarray, optional
            2D array of shape (len(indices), dim). Each row is a vector.
            If None, uses vector_fetch_fn(indices) to retrieve vectors.

        Examples
        --------
        Direct indexing with vectors:

        >>> indices = [0, 1, 2, 3, 4]
        >>> vectors = np.random.randn(5, 128)  # 5 vectors, 128 dims
        >>> lsh.index(indices, vectors)

        Indexing with fetch function:

        >>> def fetch_from_db(ids):
        ...     # Query database for vectors
        ...     return load_vectors(ids)
        >>>
        >>> lsh = LSHRS(dim=128, vector_fetch_fn=fetch_from_db)
        >>> lsh.index([100, 101, 102])  # Vectors fetched automatically

        Raises
        ------
        ValueError
            If vector shape doesn't match (n, dim) or count doesn't match indices.
        RuntimeError
            If vectors=None and no vector_fetch_fn is configured.

        Notes
        -----
        The method ensures all vectors are indexed atomically by flushing
        the buffer at the end, making them immediately searchable.
        """
        # Empty input is no-op
        if not indices:
            return

        # Fetch vectors if not provided directly
        if vectors is None:
            fetch_fn = self._require_vector_fetch_fn()
            vectors = fetch_fn(indices)

        # Validate vector array shape and type
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

        # Index each vector
        for idx, vec in zip(indices, arr):
            self.ingest(int(idx), vec)

        # Force flush to make vectors immediately searchable
        self.flush()

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
        Retrieve candidates similar to the query vector.

        This is the main search interface supporting two retrieval modes:

        1. **Top-k mode** (top_p=None): Returns k candidates with most band collisions.
           Fast, no reranking, based purely on LSH hash matches.

        2. **Top-p mode** (top_p provided): Reranks candidates by cosine similarity
           and returns enough to cover p proportion of total similarity.
           Requires vector_fetch_fn. More accurate but slower.

        The method:
            1. Hashes query vector to get signatures
            2. Retrieves candidates from matching buckets
            3. Counts band collisions per candidate
            4. Either returns top-k by collision count OR
            5. Reranks by cosine similarity and applies top-p cutoff

        Parameters
        ----------
        vector : np.ndarray
            Query vector of dimension `dim`. Will be hashed using same
            projections as indexed vectors.

        top_k : int, optional, default=10
            Maximum candidates to return. In top-p mode, acts as additional
            limit. None returns all candidates (use carefully!).

        top_p : float, optional
            Cumulative proportion threshold in (0, 1]. When provided:
            - Candidates are reranked by cosine similarity
            - Returns smallest prefix covering p proportion of similarity mass
            - Example: top_p=0.1 returns top 10% most similar
            - Requires vector_fetch_fn to be configured

        Returns
        -------
        List[int] or List[Tuple[int, float]]
            Top-k mode: List of candidate indices sorted by band collisions.
            Top-p mode: List of (index, cosine_similarity) tuples sorted by similarity.

        Examples
        --------
        Top-k retrieval (fast, no reranking):

        >>> query_vec = np.random.randn(128)
        >>> candidates = lsh.query(query_vec, top_k=20)
        >>> candidates
        [42, 17, 203, ...]  # 20 indices

        Top-p retrieval (accurate, with reranking):

        >>> scores = lsh.query(query_vec, top_p=0.05, top_k=100)
        >>> scores
        [(42, 0.95), (17, 0.93), ...]  # (index, similarity) pairs

        Unlimited results (use carefully):

        >>> all_candidates = lsh.query(query_vec, top_k=None)

        Raises
        ------
        ValueError
            If top_k <= 0, top_p not in (0,1], or vector dimension mismatch.
        RuntimeError
            If top_p provided but no vector_fetch_fn configured.

        Notes
        -----
        Band collision count correlates with similarity but isn't a direct measure.
        Use top-p mode with cosine reranking for accurate similarity scores.
        """
        # Prepare and validate query vector
        query_vector = self._prepare_vector(vector)

        # Get candidates and their band collision counts
        # More collisions = more likely to be similar
        candidate_counts = self._candidate_counts(query_vector)
        if not candidate_counts:
            return []

        # Sort by collision count (descending), then by index for stability
        ordered = sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))

        # Mode 1: Pure top-k without reranking
        if top_p is None:
            if top_k is None:
                top_k = len(ordered)
            if top_k <= 0:
                raise ValueError("top_k must be greater than zero when provided")
            return [idx for idx, _ in ordered[:top_k]]

        # Mode 2: Top-p with cosine similarity reranking
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be within the range (0, 1]")

        # Extract candidate indices for fetching
        candidate_indices = [idx for idx, _ in ordered]

        # Fetch candidate vectors for reranking
        fetch_fn = self._require_vector_fetch_fn()
        candidate_vectors = fetch_fn(candidate_indices)

        # Validate fetched vectors
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

        # Compute cosine similarities and sort
        similarities = top_k_cosine(query_vector, arr, k=len(candidate_indices))
        ordered_scores = [
            (candidate_indices[pos], score) for pos, score in similarities
        ]

        # Apply top-p cutoff (return top p% of candidates)
        limit = max(1, math.ceil(len(ordered_scores) * top_p))

        # Also apply top_k limit if specified
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be greater than zero when provided")
            limit = min(limit, top_k)

        return ordered_scores[:limit]

    def get_top_k(self, vector: np.ndarray, topk: int = 10) -> List[int]:
        """
        Convenience wrapper for pure top-k retrieval without reranking.

        Returns the k candidates with most LSH band collisions. Fast but
        approximate - collision count correlates with similarity but isn't exact.

        Parameters
        ----------
        vector : np.ndarray
            Query vector of dimension `dim`.

        topk : int, default=10
            Number of top candidates to return.

        Returns
        -------
        List[int]
            Indices of top-k candidates sorted by band collision count.

        Examples
        --------
        >>> query = np.random.randn(128)
        >>> top_10 = lsh.get_top_k(query, topk=10)
        >>> top_10
        [42, 17, 203, 91, ...]  # 10 indices

        See Also
        --------
        query : Full query interface with reranking options
        get_above_p : Top-p retrieval with cosine reranking
        """
        results = self.query(vector, top_k=topk, top_p=None)
        return list(results)  # type: ignore[return-value]

    def get_above_p(self, vector: np.ndarray, p: float = 0.95) -> CandidateScores:
        """
        Convenience wrapper for top-p retrieval with cosine similarity reranking.

        Returns candidates covering the top p proportion of similarity mass.
        More accurate than get_top_k but requires fetching candidate vectors.

        Parameters
        ----------
        vector : np.ndarray
            Query vector of dimension `dim`.

        p : float, default=0.95
            Proportion threshold in (0, 1]. Returns enough candidates to
            cover this proportion of total similarity. Lower = more selective.

        Returns
        -------
        List[Tuple[int, float]]
            List of (index, cosine_similarity) pairs sorted by similarity descending.

        Examples
        --------
        >>> query = np.random.randn(128)
        >>> # Get top 5% most similar
        >>> results = lsh.get_above_p(query, p=0.05)
        >>> results
        [(42, 0.95), (17, 0.93), (203, 0.91), ...]
        >>>
        >>> # Extract just indices
        >>> indices = [idx for idx, score in results]

        Raises
        ------
        RuntimeError
            If vector_fetch_fn is not configured.

        See Also
        --------
        query : Full query interface
        get_top_k : Fast top-k without reranking
        """
        results = self.query(vector, top_k=None, top_p=p)
        return list(results)  # type: ignore[return-value]

    # ---------------------------------------------------------------------
    # Maintenance helpers
    # ---------------------------------------------------------------------

    def delete(self, indices: Union[int, Sequence[int]]) -> None:
        """
        Remove one or more vector indices from every Redis bucket.

        This performs a hard deletion - the indices are removed from all
        LSH buckets across all bands. Use sparingly as it's expensive (must
        scan all bucket keys).

        Common use cases:
            - GDPR compliance (right to be forgotten)
            - Content moderation (remove inappropriate content)
            - Data corrections (remove outdated vectors)

        For soft deletion, consider adding a "deleted" flag in your primary
        storage and filtering results instead.

        Parameters
        ----------
        indices : int or Sequence[int]
            Vector index/indices to remove from the LSH index.

        Examples
        --------
        Delete single vector:

        >>> lsh.delete(42)

        Delete multiple vectors:

        >>> lsh.delete([10, 20, 30])

        Notes
        -----
        This operation is slow for large indices as it must scan all bucket keys.
        Consider rebuilding the index periodically instead of many deletions.
        """
        # Normalize to list for consistent handling
        if isinstance(indices, int):
            to_remove = [indices]
        else:
            to_remove = [int(idx) for idx in indices]

        # Delegate to storage layer
        self._storage.remove_indices(to_remove)

    def clear(self) -> None:
        """
        Remove all Redis keys associated with the configured prefix, deleting the LSH index stored in Redis.
        
        This deletes all hash buckets and indexed entries under the instance's Redis prefix. Projection matrices remain in memory so the index can be rebuilt after clearing. This operation is irreversible; call save_to_disk() first if you may need to restore the index.
        """
        # Ensure any buffered operations are written first
        self.flush()

        # Delete all Redis keys with our prefix
        self._storage.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Return current configuration snapshot for monitoring and debugging.

        Provides visibility into the LSH parameters and Redis configuration.
        Useful for logging, debugging, and ensuring consistent configuration
        across deployments.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary including:
            - dimension: Vector dimensionality
            - num_perm: Total hash bits
            - num_bands: Number of bands
            - rows_per_band: Bits per band
            - buffer_size: Operation buffer size
            - similarity_threshold: Target similarity
            - redis_prefix: Key namespace

        Examples
        --------
        >>> lsh = LSHRS(dim=128, num_bands=20, rows_per_band=6)
        >>> lsh.stats()
        {
            'dimension': 128,
            'num_perm': 120,
            'num_bands': 20,
            'rows_per_band': 6,
            'buffer_size': 8912,
            'similarity_threshold': 0.5,
            'redis_prefix': 'lsh'
        }
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
        Persist configuration and projection matrices to disk (Secure JSON/NumPy format).

        Saves the LSH configuration to a JSON file and projection matrices to a compressed
        NumPy archive (.npz). This directory-based format is secure and portable.

        This enables:
            - Consistent hashing across restarts
            - Sharing projections between systems
            - Backup of LSH configuration
            - Migration to different Redis instances

        Parameters
        ----------
        path : str or Path
            Output directory path. Will be created if it doesn't exist.

        Examples
        --------
        >>> lsh = LSHRS(dim=128, num_bands=20, rows_per_band=6)
        >>> lsh.save_to_disk("model_v1")
        >>>
        >>> # Later, restore
        >>> restored = LSHRS.load_from_disk("model_v1")
        """
        # Ensure buffer is empty before saving
        self.flush()

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata (configuration)
        # NOTE: The redis_config contains the password, which must not be persisted.
        # Create a shallow copy and redact the password key before saving.
        sanitized_redis_config = self._redis_config.copy()
        if "password" in sanitized_redis_config:
            sanitized_redis_config["password"] = "<REDACTED>"

        metadata = {
            "version": "0.1.1a4",
            "config": self._config,
            "redis_config": sanitized_redis_config,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 2. Save projections (efficient binary format)
        # We save them as a list of arrays in a single compressed file
        np.savez_compressed(
            output_dir / "projections.npz",
            *self._hasher.projections
        )

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
        Restore an LSHRS instance from a directory saved via save_to_disk().

        Reconstructs the LSH hasher with identical projection matrices from the
        secure JSON/NumPy storage format.

        Parameters
        ----------
        path : str or Path
            Path to the directory containing metadata.json and projections.npz.

        redis_config : dict, optional
            Override saved Redis configuration.

        vector_fetch_fn : callable, optional
            Function for fetching vectors (required for reranking).

        storage : RedisStorage, optional
            Pre-configured storage instance. Overrides redis_config.

        Returns
        -------
        LSHRS
            Restored instance with same projection matrices as original.

        Raises
        ------
        FileNotFoundError
            If directory or required files don't exist.
        ValueError
            If version mismatch or corrupted data.
        """
        input_dir = Path(path)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # 1. Load metadata
        with open(input_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Basic version check (forward compatibility warning could go here)
        # stored_version = metadata.get("version")

        config = metadata["config"]
        stored_redis = metadata["redis_config"].copy()

        # NOTE: The stored redis_config may have a redacted password.
        # If a password is required, the user must supply it via the redis_config
        # parameter during load, as credentials are not persistently stored.
        # Apply Redis config overrides
        if redis_config:
            stored_redis.update(redis_config)

        # Reconstruct instance
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

        # 2. Load projections
        with np.load(input_dir / "projections.npz") as data:
            # Files in .npz are named arr_0, arr_1, ... by default when using *args
            instance._hasher.projections = [
                data[f"arr_{i}"].astype(np.float32)
                for i in range(len(data.files))
            ]

        return instance

    # ---------------------------------------------------------------------
    # Pickle protocol
    # ---------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """
        Produce a minimal, pickle-friendly representation of the instance suitable for serialization.
        
        This representation includes the LSH configuration, Redis connection configuration (for reference), and the projection matrices required to reconstruct the hasher. It deliberately excludes live resources and transient state such as the active Redis connection, the configured vector_fetch_fn, and any buffered operations.
        
        Returns:
            state (dict): Serializable mapping with keys:
                - "config": copy of the instance configuration.
                - "redis_config": copy of the Redis connection/configuration.
                - "projections": list of projection matrices as float32 NumPy arrays.
        """
        # Ensure buffer is empty (transient state shouldn't be persisted)
        self.flush()

        return {
            "config": self._config.copy(),
            "redis_config": self._redis_config.copy(),
            "projections": [
                np.asarray(matrix, dtype=np.float32)
                for matrix in self._hasher.projections
            ],
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore instance from pickled state.

        Used by pickle.loads() to reconstruct the instance. Creates a new
        instance with saved configuration and replaces its state.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from __getstate__().
        """
        # Create new instance with saved configuration
        restored = self.__class__(
            dim=state["config"]["dim"],
            num_perm=state["config"]["num_perm"],
            num_bands=state["config"]["num_bands"],
            rows_per_band=state["config"]["rows_per_band"],
            similarity_threshold=state["config"]["similarity_threshold"],
            buffer_size=state["config"]["buffer_size"],
            vector_fetch_fn=None,  # Not persisted
            redis_host=state["redis_config"]["host"],
            redis_port=state["redis_config"]["port"],
            redis_db=state["redis_config"]["db"],
            redis_password=state["redis_config"]["password"],
            redis_prefix=state["redis_config"]["prefix"],
            decode_responses=state["redis_config"]["decode_responses"],
            seed=state["config"]["seed"],
        )

        # Replace our __dict__ with restored instance's __dict__
        self.__dict__ = restored.__dict__

        # Restore saved projection matrices
        self._hasher.projections = [
            np.asarray(matrix, dtype=np.float32) for matrix in state["projections"]
        ]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input vector to expected format.

        Ensures vector is:
            - Converted to numpy array
            - Cast to float32 (memory efficient, sufficient precision)
            - Reshaped to 1D
            - Has correct dimension

        Parameters
        ----------
        vector : np.ndarray
            Input vector (can be any array-like).

        Returns
        -------
        np.ndarray
            Normalized 1D float32 array of shape (dim,).

        Raises
        ------
        ValueError
            If vector dimension doesn't match configured dim.
        """
        # Convert to float32 and ensure 1D
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)

        # Validate dimension
        if arr.shape[0] != self._dim:
            raise ValueError(
                f"Vector must have dimension {self._dim}; received {arr.shape[0]}"
            )

        # Zero vector check - norm is undefined/zero, cannot be used for cosine similarity
        if np.allclose(arr, 0.0, atol=1e-8):
            raise ValueError(
                "Cannot index zero vector - norm undefined. Check embeddings for corruption."
            )

        return arr

    def _candidate_counts(self, query_vector: np.ndarray) -> Dict[int, int]:
        """
        Compute per-candidate counts of LSH band collisions for a prepared query vector.
        
        Counts how many bands produced the same bucket as the query; larger counts indicate more band matches and therefore stronger candidate signals.
        
        Parameters:
            query_vector (np.ndarray): Prepared query vector of shape (dim,).
        
        Returns:
            Dict[int, int]: Mapping from candidate index to number of band collisions.
        """
        # Hash query vector to get signatures for all bands
        signatures = self._hasher.hash_vector(query_vector)

        # Count collisions across all bands
        counts: Dict[int, int] = {}
        for band_id, hash_val in enumerate(signatures):
            # Get all indices in this band's bucket
            for candidate in self._storage.get_bucket(band_id, hash_val):
                # Increment collision count
                counts[candidate] = counts.get(candidate, 0) + 1

        return counts

    def _enqueue_operations(
        self,
        index: int,
        signatures: Iterable[bytes],
    ) -> None:
        """
        Queue bucket-add operations for a vector's per-band signatures into the internal buffer for later batch execution.
        
        Parameters:
            index (int): Non-negative identifier of the vector being indexed.
            signatures (Iterable[bytes]): Per-band hash signatures (one entry per band).
        """
        with self._buffer_lock:
            # Create operation for each band
            for band_id, hash_val in enumerate(signatures):
                self._buffer.append((band_id, hash_val, int(index)))
        # Note: Buffer is flushed lazily to leverage pipelining

    def _flush_buffer_if_needed(self) -> None:
        """
        Flush buffer to Redis if it reaches the configured size limit.

        Prevents unbounded memory growth during continuous ingestion.
        """
        should_flush = False
        with self._buffer_lock:
            if len(self._buffer) >= self._buffer_size:
                should_flush = True

        if should_flush:
            self.flush()

    def _require_vector_fetch_fn(self) -> VectorFetchFn:
        """
        Return the configured vector fetch function or raise if none is configured.
        
        Returns:
            VectorFetchFn: The configured function used to fetch vectors by indices.
        
        Raises:
            RuntimeError: If `vector_fetch_fn` is not configured.
        """
        if self._vector_fetch_fn is None:
            raise RuntimeError(
                "vector_fetch_fn must be supplied for operations requiring reranking"
            )
        return self._vector_fetch_fn

    def _resolve_loader(self, format: str) -> Loader:
        """
        Map format string to appropriate data loader function.

        Dynamically imports loader modules to avoid unnecessary dependencies.

        Parameters
        ----------
        format : str
            Format identifier (case-insensitive).

        Returns
        -------
        Loader
            Function that yields (indices, vectors) batches.

        Raises
        ------
        ValueError
            If format is not supported.
        ImportError
            If required dependencies for format are not installed.
        """
        normalized = format.lower()

        # PostgreSQL loader
        if normalized in {"postgres", "pg"}:
            from lshrs.io.postgres import iter_postgres_vectors

            return iter_postgres_vectors

        # Parquet loader
        if normalized in {"parquet", "pq"}:
            from lshrs.io.parquet import iter_parquet_vectors

            return iter_parquet_vectors

        raise ValueError(f"Unsupported signature creation format '{format}'")


# Backwards compatibility alias for earlier versions that used lowercase naming
# Allows: from lshrs import lshrs (old style) or LSHRS (new style)
lshrs = LSHRS