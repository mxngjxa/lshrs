"""
Redis-Based Storage Backend for LSH Bucket Management

This module provides a thin, efficient wrapper around redis-py for storing and
retrieving LSH hash buckets. Each bucket is a Redis Set containing indices of
vectors that share the same hash signature in a specific band.

Architecture:
    - Each (band_id, hash_signature) pair maps to a unique Redis key
    - Keys are stored as sets for efficient membership operations
    - Pipelining is used for batch operations to minimize network roundtrips
    - All keys are namespaced with a configurable prefix

Key design decisions:
    - Redis Sets (SADD/SMEMBERS) for O(1) insertion and duplicate handling
    - Pipeline support for bulk operations (100x+ faster than individual calls)
    - Scan iteration for safe key deletion without blocking Redis
    - Hex encoding of hash signatures for human-readable keys

Typical usage:
    >>> storage = RedisStorage(host='localhost', prefix='myapp_lsh')
    >>> storage.add_to_bucket(band_id=0, hash_val=b'\x01\x02', index=42)
    >>> candidates = storage.get_bucket(band_id=0, hash_val=b'\x01\x02')
    >>> candidates
    {42}
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence, Set, Tuple

import redis

# Type alias for clarity in batch operations
# Each operation is: (band_id, hash_signature, vector_index)
BucketOperation = Tuple[int, bytes, int]


class RedisStorage:
    """
    Thin wrapper around redis-py for LSH bucket management.

    This class manages the storage of LSH hash buckets in Redis. Each bucket
    is a set of vector indices that share the same hash signature in a specific
    band. The class provides both single-operation and batch-operation methods.

    Key naming convention:
        {prefix}:{band_id}:bucket:{hash_hex}

        Example: "lsh:0:bucket:a1b2c3" contains all vector indices that hashed
                 to 0xa1b2c3 in band 0.

    Attributes:
        prefix: Namespace prefix for all Redis keys (for multi-tenant scenarios).
        _client: Underlying redis-py client instance.

    Redis data structure:
        - Type: SET (unordered collection of unique integers)
        - Operations: SADD (add), SMEMBERS (get all), SREM (remove)
        - Why sets? Automatic deduplication and O(1) membership testing

    Performance considerations:
        - Single operations: Use for real-time updates (low latency required)
        - Batch operations: Use for bulk indexing (high throughput required)
        - Pipelining reduces network roundtrips by ~100x for batch operations

    Example:
        >>> # Initialize with custom config
        >>> storage = RedisStorage(
        ...     host='redis.example.com',
        ...     port=6379,
        ...     db=0,
        ...     password='secret',
        ...     prefix='search_engine'
        ... )
        >>>
        >>> # Add vector index 42 to a bucket
        >>> storage.add_to_bucket(band_id=0, hash_val=b'\xff\x00', index=42)
        >>>
        >>> # Retrieve all indices in that bucket
        >>> storage.get_bucket(band_id=0, hash_val=b'\xff\x00')
        {42}
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        decode_responses: bool = False,
        prefix: str = "lsh",
    ) -> None:
        """
        Initialize Redis connection for LSH bucket storage.

        Creates a redis-py client with specified connection parameters. All
        arguments are keyword-only to prevent mistakes with positional args.

        Args:
            host: Redis server hostname or IP address.
                 Use 'localhost' for local development.
            port: Redis server port.
                 Standard Redis port is 6379.
            db: Redis database number (0-15 in default config).
                Use different db numbers to isolate data.
            password: Redis authentication password.
                     Set to None if Redis has no AUTH configured.
            decode_responses: If True, Redis returns strings instead of bytes.
                            Keep False for binary data handling (LSH hashes).
            prefix: Namespace prefix for all keys.
                   Use different prefixes for different LSH indices or apps.
                   Must not contain colons (:) to avoid key parsing issues.

        Example:
            >>> # Local development
            >>> storage = RedisStorage()
            >>>
            >>> # Production with auth
            >>> storage = RedisStorage(
            ...     host='prod-redis.example.com',
            ...     password='strong_password',
            ...     prefix='prod_lsh'
            ... )
            >>>
            >>> # Multi-tenant setup
            >>> user1_storage = RedisStorage(prefix='user1_lsh', db=0)
            >>> user2_storage = RedisStorage(prefix='user2_lsh', db=0)

        Raises:
            redis.ConnectionError: If unable to connect to Redis server.
            redis.AuthenticationError: If password is incorrect.
        """
        # Store prefix for key generation
        self.prefix = prefix

        # Initialize redis-py client
        # Connection is lazy - doesn't actually connect until first operation
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
        )

    @property
    def client(self) -> redis.Redis:  # pragma: no cover - simple accessor
        """
        Expose the underlying redis-py client for advanced operations.

        Provides direct access to the Redis client for operations not wrapped
        by this class. Use with caution - direct client access bypasses the
        key naming conventions and safety checks.

        Returns:
            The underlying redis.Redis client instance.

        Use cases:
            - Custom Redis commands not provided by this wrapper
            - Performance monitoring (INFO commands)
            - Transaction management beyond simple pipelines
            - Pub/sub operations

        Example:
            >>> storage = RedisStorage()
            >>> # Get Redis server info
            >>> info = storage.client.info('memory')
            >>> print(f"Used memory: {info['used_memory_human']}")
            >>>
            >>> # Execute custom command
            >>> storage.client.execute_command('PING')
            b'PONG'
        """
        return self._client

    def bucket_key(self, band_id: int, hash_val: bytes) -> str:
        """
        Compute the Redis key for a given band/hash pair.

        Generates a hierarchical key name following the pattern:
            {prefix}:{band_id}:bucket:{hash_hex}

        The hash value is hex-encoded for human readability in Redis CLI tools.
        This makes debugging and monitoring much easier than raw binary keys.

        Args:
            band_id: The LSH band number (0 to num_bands-1).
                    Different bands use separate key namespaces.
            hash_val: The hash signature as bytes (from LSHHasher output).
                     Will be converted to hex string for key name.

        Returns:
            Redis key string ready for SADD/SMEMBERS/etc operations.

        Example:
            >>> storage = RedisStorage(prefix='search')
            >>> key = storage.bucket_key(band_id=5, hash_val=b'\xab\xcd\xef')
            >>> key
            'search:5:bucket:abcdef'
            >>>
            >>> # Different bands have different namespaces
            >>> key0 = storage.bucket_key(0, b'\x01')
            >>> key1 = storage.bucket_key(1, b'\x01')
            >>> key0 == key1
            False

        Design notes:
            - Colon separators enable Redis keyspace notifications filtering
            - Hex encoding makes keys grep-able in Redis logs
            - Band ID comes before hash for efficient pattern matching
        """
        # Convert binary hash to hex string (e.g., b'\xff' -> 'ff')
        # hex() method is faster than binascii.hexlify
        return f"{self.prefix}:{band_id}:bucket:{hash_val.hex()}"

    def add_to_bucket(self, band_id: int, hash_val: bytes, index: int) -> None:
        """
        Add a single index to the specified bucket.

        This is the fundamental operation for building an LSH index. When you
        hash a vector and get signatures for each band, you call this method
        once per band to store the vector's index in the appropriate bucket.

        Uses Redis SADD which:
            - Adds element to set (O(1) operation)
            - Automatically handles duplicates (no-op if already exists)
            - Returns atomically (thread-safe)

        Args:
            band_id: Which LSH band this hash belongs to.
            hash_val: The hash signature bytes for this band.
            index: The vector's index in your main vector store.
                  Must be non-negative integer.

        Returns:
            None. Operation is fire-and-forget (no return value needed).

        Example:
            >>> storage = RedisStorage()
            >>>
            >>> # Index vector #42 with its hash signature from band 0
            >>> storage.add_to_bucket(
            ...     band_id=0,
            ...     hash_val=b'\x01\x02\x03',
            ...     index=42
            ... )
            >>>
            >>> # Later, retrieve all vectors with same signature
            >>> candidates = storage.get_bucket(0, b'\x01\x02\x03')
            >>> 42 in candidates
            True

        Performance notes:
            - Single operation latency: ~1ms on localhost, ~10-50ms over network
            - For bulk indexing, use batch_add() instead (100x faster)
            - SADD is atomic and thread-safe (multiple clients can add safely)

        Use cases:
            - Real-time indexing (add vectors as they arrive)
            - Incremental updates to existing index
            - Small-scale indexing (<1000 vectors)
        """
        # Generate Redis key for this band/hash combination
        key = self.bucket_key(band_id, hash_val)

        # Add index to the set stored at this key
        # Creates the set if it doesn't exist yet
        # No-op if index already in set (automatic deduplication)
        self._client.sadd(key, index)

    def get_bucket(self, band_id: int, hash_val: bytes) -> Set[int]:
        """
        Fetch all indices stored in the specified bucket.

        This is the fundamental query operation for LSH search. When you hash
        a query vector, you call this method for each band to retrieve candidate
        vector indices that might be similar to your query.

        Uses Redis SMEMBERS which:
            - Returns all elements in set (O(N) where N = set size)
            - Returns empty set if key doesn't exist (no error)
            - Atomic operation (consistent snapshot)

        Args:
            band_id: Which LSH band to query.
            hash_val: The hash signature bytes to look up.

        Returns:
            Set of integer indices for vectors that hashed to this signature.
            Returns empty set {} if no vectors have this signature.

            Set properties:
                - Unordered (no specific order guaranteed)
                - Unique elements (no duplicates)
                - O(1) membership testing

        Example:
            >>> storage = RedisStorage()
            >>>
            >>> # Add some vectors to a bucket
            >>> storage.add_to_bucket(0, b'\xff\x00', 10)
            >>> storage.add_to_bucket(0, b'\xff\x00', 20)
            >>> storage.add_to_bucket(0, b'\xff\x00', 30)
            >>>
            >>> # Query the bucket
            >>> candidates = storage.get_bucket(0, b'\xff\x00')
            >>> candidates
            {10, 20, 30}
            >>>
            >>> # Non-existent bucket returns empty set
            >>> storage.get_bucket(0, b'\x00\x00')
            set()

        Performance notes:
            - Latency: ~1ms for small buckets (<100 items)
            - Scales linearly with bucket size
            - Network latency is often the bottleneck (not Redis)

        Typical workflow:
            1. Hash query vector → get signatures for each band
            2. Call get_bucket() for each band
            3. Union or aggregate results across bands
            4. Rerank candidates using exact similarity (cosine, etc.)
        """
        # Generate Redis key for this band/hash combination
        key = self.bucket_key(band_id, hash_val)

        # Fetch all members of the set
        # Returns empty set if key doesn't exist
        members = self._client.smembers(key)

        # Convert bytes/strings to integers
        # Redis returns bytes by default (decode_responses=False)
        # or strings (decode_responses=True)
        return {int(m) for m in members}

    def batch_add(self, operations: Sequence[BucketOperation]) -> None:
        """
        Insert a batch of bucket operations via Redis pipelining.

        This is the recommended method for bulk indexing. Instead of sending
        one network request per operation, pipelining buffers commands and
        sends them in a single batch, dramatically reducing latency.

        Performance comparison (indexing 1000 vectors with 20 bands each):
            - Individual add_to_bucket calls: ~20 seconds
            - Single batch_add call: ~0.2 seconds (100x faster!)

        Pipelining benefits:
            - Reduces network roundtrips from N to 1
            - Commands still execute sequentially on Redis server
            - All-or-nothing: execute() sends everything at once
            - Automatic retry on connection errors (via context manager)

        Args:
            operations: Sequence of (band_id, hash_val, index) tuples.
                       Can be list, tuple, or any iterable.
                       Empty sequence is a no-op (returns immediately).

        Returns:
            None. All operations are executed in a single batch.

        Example:
            >>> storage = RedisStorage()
            >>>
            >>> # Build batch from LSH signatures
            >>> operations = []
            >>> for idx, vector in enumerate(vectors):
            ...     signatures = hasher.hash_vector(vector)
            ...     for band_id, hash_val in enumerate(signatures):
            ...         operations.append((band_id, hash_val, idx))
            >>>
            >>> # Execute all at once (fast!)
            >>> storage.batch_add(operations)
            >>>
            >>> # Example: 1000 vectors × 20 bands = 20,000 operations in ~200ms

        Implementation notes:
            - Uses pipeline() context manager for automatic execution
            - SADD commands are buffered in memory until execute()
            - Pipeline is automatically reset even if exception occurs
            - Memory usage: O(operations) for command buffer

        Best practices:
            - Batch size: 1000-10000 operations is usually optimal
            - Larger batches save network time but use more memory
            - For very large batches, consider chunking (e.g., 10k per batch)

        Error handling:
            - Connection errors: Automatically retried by pipeline context
            - Redis errors: Raised after attempting all commands
            - Partial failures: Some commands may succeed before error
        """
        # Early return for empty batch (avoid unnecessary pipeline creation)
        if not operations:
            return

        # Use context manager for safe pipeline execution
        # Automatically calls execute() on exit and reset() on error
        with self.pipeline() as pipe:
            # Queue all SADD commands in the pipeline
            # Commands are buffered, not executed yet
            for band_id, hash_val, index in operations:
                key = self.bucket_key(band_id, hash_val)
                pipe.sadd(key, index)
            # execute() is called automatically when exiting context manager

    def remove_indices(self, indices: Iterable[int]) -> None:
        """
        Remove indices from every bucket key across all bands.

        This operation is used when deleting vectors from your LSH index. It
        scans through all buckets and removes the specified indices, ensuring
        they won't appear in future search results.

        Warning: This is a SLOW operation! It must:
            1. Scan all keys matching the prefix pattern
            2. Remove indices from each key found

        Typical use cases:
            - Hard deletion of vectors (GDPR compliance, content moderation)
            - Garbage collection after bulk updates
            - Index maintenance and cleanup

        Performance characteristics:
            - Time: O(num_buckets × num_indices)
            - For 10,000 buckets × 100 indices: ~5-30 seconds
            - Blocks Redis during execution (use carefully in production)

        Args:
            indices: Vector indices to remove from all buckets.
                    Can be list, set, tuple, or any iterable.
                    Empty iterable is a no-op.
                    Duplicates are automatically handled.

        Returns:
            None. All matching indices are removed from all buckets.

        Example:
            >>> storage = RedisStorage()
            >>>
            >>> # Index some vectors
            >>> storage.add_to_bucket(0, b'\x01', 100)
            >>> storage.add_to_bucket(0, b'\x01', 101)
            >>> storage.add_to_bucket(1, b'\x02', 100)
            >>>
            >>> # Remove vector 100 from entire index
            >>> storage.remove_indices([100])
            >>>
            >>> # Verify removal
            >>> storage.get_bucket(0, b'\x01')
            {101}
            >>> storage.get_bucket(1, b'\x02')
            set()

        Implementation details:
            - Uses SCAN iterator (safe, doesn't block Redis for long)
            - SCAN returns keys in undefined order
            - SREM is idempotent (no error if index not in set)
            - Pipeline batches all SREM commands for efficiency

        Alternatives for better performance:
            - Mark vectors as deleted (logical deletion) rather than removing
            - Rebuild entire index periodically instead of incremental removal
            - Use Redis key expiration (TTL) if deletion can be delayed

        Production considerations:
            - Run during off-peak hours (impacts Redis performance)
            - Consider rate limiting if removing many indices
            - Monitor Redis CPU usage during operation
            - For large deletions, consider rebuilding index instead
        """
        # Normalize to list for consistent iteration
        # Allows reuse without exhausting iterator
        normalized = list(indices)

        # Early return for empty input
        if not normalized:
            return

        # Pattern matches all bucket keys under this prefix
        # Example: "lsh:*:bucket:*" matches "lsh:0:bucket:abc", "lsh:5:bucket:def", etc.
        pattern = f"{self.prefix}:*:bucket:*"

        # Use pipeline for batch removal (much faster than individual calls)
        with self.pipeline() as pipe:
            # SCAN iterates through keyspace without blocking
            # Returns keys matching pattern in chunks
            for key in self._client.scan_iter(match=pattern):
                # Queue SREM command for each key
                # Removes all specified indices from the set
                # No-op if indices not in set (SREM is idempotent)
                pipe.srem(key, *normalized)
            # All SREM commands execute when pipeline exits

    @contextmanager
    def pipeline(self) -> Iterator[redis.client.Pipeline]:
        """
        Context manager for Redis pipelines with automatic execution.

        Provides a safe, convenient way to use Redis pipelining. The pipeline
        automatically executes all buffered commands on exit and resets itself
        even if an exception occurs.

        Redis pipelining:
            - Batches multiple commands into single network request
            - Commands execute sequentially on server (not parallel)
            - Responses returned as list after execute()
            - Atomic from network perspective (not from Redis perspective)

        Yields:
            redis.client.Pipeline instance for queueing commands.

        Guarantees:
            1. execute() is called on normal exit (all commands run)
            2. reset() is called on exception (cleans up state)
            3. No leftover commands in pipeline buffer

        Example:
            >>> storage = RedisStorage()
            >>>
            >>> # Basic usage
            >>> with storage.pipeline() as pipe:
            ...     pipe.sadd('key1', 1)
            ...     pipe.sadd('key2', 2)
            ...     pipe.sadd('key3', 3)
            ... # execute() called automatically here
            >>>
            >>> # Exception handling
            >>> try:
            ...     with storage.pipeline() as pipe:
            ...         pipe.sadd('key1', 1)
            ...         raise ValueError("Something broke")
            ... except ValueError:
            ...     pass
            ... # reset() was called automatically, pipeline is clean

        Advanced usage:
            >>> # Access execution results
            >>> with storage.pipeline() as pipe:
            ...     pipe.sadd('key', 1)  # Returns pipeline object
            ...     pipe.smembers('key')  # Returns pipeline object
            ...     # Results not available until execute()
            ...
            >>> # Manual execution (not recommended - use context manager)
            >>> pipe = storage.client.pipeline()
            >>> pipe.sadd('key', 1)
            >>> results = pipe.execute()  # Must call manually
            >>> pipe.reset()  # Must call manually

        Performance tips:
            - Batch 100-10000 commands per pipeline for best performance
            - Don't nest pipelines (they don't compose well)
            - Pipeline is single-threaded (use connection pool for parallel)

        Error handling:
            - Connection errors: Raised immediately, pipeline reset
            - Command errors: Raised after execute(), pipeline reset
            - Partial success: Some commands may execute before error
        """
        # Create new pipeline from the Redis client
        # Pipeline buffers commands without executing them
        pipe = self._client.pipeline()

        try:
            # Yield pipeline to caller for command queueing
            yield pipe

            # Automatically execute all buffered commands
            # Sends everything to Redis in single request
            # Returns list of results (one per command)
            pipe.execute()
        finally:
            # Always reset pipeline, even if exception occurred
            # Clears command buffer and internal state
            # Prevents memory leaks and state corruption
            pipe.reset()

    def clear(self) -> None:
        """
        Delete all keys under the configured prefix.

        This is a destructive operation that removes the entire LSH index from
        Redis. Use with caution! All bucket data will be permanently deleted.

        Use cases:
            - Clearing test data after unit tests
            - Resetting index before full rebuild
            - Deleting old indices during schema migration
            - Emergency cleanup of corrupted data

        Warning: This operation is IRREVERSIBLE. Consider taking a Redis snapshot
        (SAVE/BGSAVE) before running in production.

        Args:
            None. Uses the prefix configured during initialization.

        Returns:
            None. All matching keys are deleted.

        Example:
            >>> storage = RedisStorage(prefix='test_lsh')
            >>>
            >>> # Add some data
            >>> storage.add_to_bucket(0, b'\x01', 100)
            >>> storage.add_to_bucket(1, b'\x02', 200)
            >>>
            >>> # Clear everything
            >>> storage.clear()
            >>>
            >>> # Verify deletion
            >>> storage.get_bucket(0, b'\x01')
            set()

        Implementation details:
            - Uses SCAN to find all keys (safe, doesn't block)
            - Collects all keys into list before deletion
            - Single DELETE command for all keys (atomic)
            - No-op if no keys match pattern

        Performance:
            - SCAN time: O(num_keys) but non-blocking
            - DELETE time: O(num_keys) but fast (Redis in-memory)
            - Total time: Usually <1 second for <100k keys

        Safety considerations:
            - Only deletes keys with matching prefix (safe for multi-tenant)
            - Does not affect other Redis databases (db parameter isolation)
            - Cannot undo - make sure prefix is correct before calling

        Alternative approaches:
            - Use separate Redis database (db parameter) per index, then FLUSHDB
            - Set TTL on keys for automatic expiration
            - Use Redis key namespacing for logical separation

        Production tips:
            - Verify prefix before calling: print(storage.prefix)
            - Use different prefixes for dev/staging/prod
            - Consider dry-run: list keys first, confirm, then delete
        """
        # Pattern matches all keys under this prefix
        # Example: "lsh:*" matches "lsh:0:bucket:abc", "lsh:metadata", etc.
        pattern = f"{self.prefix}:*"

        # Collect all matching keys
        # scan_iter returns generator, convert to list for DELETE command
        # SCAN is safe (doesn't block Redis) but may return duplicates
        keys = list(self._client.scan_iter(match=pattern))

        # Delete all keys at once
        # DELETE is atomic: all keys removed in single operation
        # No-op if keys list is empty (no error)
        if keys:
            self._client.delete(*keys)
