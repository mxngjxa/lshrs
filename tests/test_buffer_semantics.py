# import pytest
import numpy as np
from typing import List, Set, Tuple, Any
from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage


class MockStorage(RedisStorage):
    def __init__(self):
        """
        Initialize an in-memory mock storage for testing LSH buffering and bucket lookups.

        Creates:
        - self.batches: a list that records appended batches of (band_id, hash_val, index) operations.
        - self.data: a mapping from (band_id, hash_val) tuples to sets of indices representing bucket contents.
        """
        self.batches: List[Any] = []
        self.data: dict = {}

    def batch_add(self, operations: List[Tuple[int, bytes, int]]) -> None:
        """
        Record a batch of indexing operations and update in-memory buckets for testing.

        Parameters:
            operations (List[Tuple[int, bytes, int]]): Sequence of tuples (band_id, hash_val, index) representing indexing operations to append and apply to the in-memory storage.

        Description:
            Appends the given operations list to the recorded batches and, for each tuple, adds the provided index to the set associated with the (band_id, hash_val) key in the in-memory `data` mapping. This method is intended for use in tests to simulate batched writes and support in-memory queries.
        """
        self.batches.append(operations)
        for band_id, hash_val, index in operations:
            key = (band_id, hash_val)
            if key not in self.data:
                self.data[key] = set()
            self.data[key].add(index)

    def get_bucket(self, band_id: int, hash_val: bytes) -> Set[int]:
        """
        Return the set of vector indices stored for the given band and hash.

        Parameters:
            band_id (int): Band identifier used in the LSH banding.
            hash_val (bytes): Bucket hash value.

        Returns:
            Set[int]: Indices assigned to the (band_id, hash_val) bucket; empty set if the bucket does not exist.
        """
        return self.data.get((band_id, hash_val), set())

    def close(self) -> None:
        """
        Close the storage without performing any action.

        This no-op method exists to satisfy the storage interface; it does not flush buffers or release resources.
        """
        pass


def test_single_ingest_not_immediately_queryable():
    """Test that single ingestion is buffered and not immediately searchable."""
    storage = MockStorage()
    # Large buffer ensures no auto-flush
    lsh = LSHRS(
        dim=4,
        num_bands=2,
        rows_per_band=2,
        num_perm=4,
        buffer_size=100,
        storage=storage,
    )

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Ingest one vector
    lsh.ingest(0, vec)

    # Buffer should not have flushed yet
    assert len(storage.batches) == 0

    # Query should return empty results because storage is empty
    results = lsh.query(vec, top_k=1)
    assert results == []

    # Manually flush
    lsh.flush()

    # Now storage should have data
    assert len(storage.batches) == 1

    # Query should work
    results = lsh.query(vec, top_k=1)
    assert results == [0]


def test_batch_index_auto_flushes():
    """Test that batch indexing forces a flush at the end."""
    storage = MockStorage()
    lsh = LSHRS(
        dim=4,
        num_bands=2,
        rows_per_band=2,
        num_perm=4,
        buffer_size=100,
        storage=storage,
    )

    vecs = np.eye(4, dtype=np.float32)
    indices = [0, 1, 2, 3]

    # Batch index
    lsh.index(indices, vecs)

    # Should have flushed automatically
    assert len(storage.batches) == 1
    # 4 vectors * 2 bands = 8 operations
    assert len(storage.batches[0]) == 8

    # Immediately queryable
    results = lsh.query(vecs[0], top_k=1)
    assert results == [0]


def test_buffer_flush_on_full():
    """Test that buffer flushes automatically when capacity is reached."""
    storage = MockStorage()
    # buffer_size=4. Each vector adds 2 ops (2 bands).
    # 2 vectors = 4 ops, which should trigger flush.
    lsh = LSHRS(
        dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=4, storage=storage
    )

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Ingest 1 (2 ops)
    lsh.ingest(0, vec)
    assert len(storage.batches) == 0

    # Ingest 2 (Total 4 ops) -> Should trigger flush
    lsh.ingest(1, vec)
    assert len(storage.batches) == 1
    assert len(storage.batches[0]) == 4


def test_close_flushes_buffer():
    """Test that closing the LSH instance flushes any pending operations."""
    storage = MockStorage()
    lsh = LSHRS(
        dim=4,
        num_bands=2,
        rows_per_band=2,
        num_perm=4,
        buffer_size=100,
        storage=storage,
    )

    lsh.ingest(0, np.zeros(4, dtype=np.float32) + 1)  # Non-zero vector
    assert len(storage.batches) == 0

    lsh.close()
    assert len(storage.batches) == 1
