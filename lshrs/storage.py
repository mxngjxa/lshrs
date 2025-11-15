from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, List, Sequence, Set, Tuple

import redis

BucketOperation = Tuple[int, bytes, int]


class RedisStorage:
    """Thin wrapper around redis-py for LSH bucket management."""

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
        self.prefix = prefix
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
        )

    @property
    def client(self) -> redis.Redis:  # pragma: no cover - simple accessor
        return self._client

    def bucket_key(self, band_id: int, hash_val: bytes) -> str:
        """Compute the Redis key for a given band/hash pair."""
        return f"{self.prefix}:{band_id}:bucket:{hash_val.hex()}"

    def add_to_bucket(self, band_id: int, hash_val: bytes, index: int) -> None:
        """Add a single index to the specified bucket."""
        key = self.bucket_key(band_id, hash_val)
        self._client.sadd(key, index)

    def get_bucket(self, band_id: int, hash_val: bytes) -> Set[int]:
        """Fetch all indices stored in the specified bucket."""
        key = self.bucket_key(band_id, hash_val)
        members = self._client.smembers(key)
        return {int(m) for m in members}

    def batch_add(self, operations: Sequence[BucketOperation]) -> None:
        """Insert a batch of bucket operations via Redis pipelining."""
        if not operations:
            return
        with self.pipeline() as pipe:
            for band_id, hash_val, index in operations:
                key = self.bucket_key(band_id, hash_val)
                pipe.sadd(key, index)

    def remove_indices(self, indices: Iterable[int]) -> None:
        """Remove indices from every bucket key."""
        normalized = list(indices)
        if not normalized:
            return

        pattern = f"{self.prefix}:*:bucket:*"
        with self.pipeline() as pipe:
            for key in self._client.scan_iter(match=pattern):
                pipe.srem(key, *normalized)

    @contextmanager
    def pipeline(self) -> Iterator[redis.client.Pipeline]:
        """Context manager for Redis pipelines with automatic execution."""
        pipe = self._client.pipeline()
        try:
            yield pipe
            pipe.execute()
        finally:
            pipe.reset()

    def clear(self) -> None:
        """Delete all keys under the configured prefix."""
        pattern = f"{self.prefix}:*"
        keys = list(self._client.scan_iter(match=pattern))
        if keys:
            self._client.delete(*keys)