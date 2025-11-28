import threading
import time
from typing import Any, List
# from unittest.mock import MagicMock

import numpy as np
# import pytest

from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage


class MockStorage(RedisStorage):
    def __init__(self):
        """
        Initialize mock storage state for testing.

        Creates and initializes internal counters and collections used to record batched operations and a threading.Lock to synchronize access across threads:
        - batch_add_calls: counts how many times batch_add was invoked.
        - operations: list collecting operations passed to batch_add.
        - lock: threading.Lock protecting shared state.
        """
        self.batch_add_calls = 0
        self.operations = []
        self.lock = threading.Lock()

    def batch_add(self, operations: List[Any]) -> None:
        """
        Record a batch of storage operations in a thread-safe in-memory mock and simulate IO latency.

        Acquires the instance lock, increments the internal batch_add_calls counter, appends the given operations to the stored operations list, and pauses briefly to simulate I/O delay.

        Parameters:
            operations (List[Any]): The sequence of operations to append to the mock storage's internal operations list.
        """
        with self.lock:
            self.batch_add_calls += 1
            self.operations.extend(operations)
            # Simulate some IO latency
            time.sleep(0.001)

    def get_bucket(self, band_id: int, hash_val: bytes):
        """
        Return the set of stored item indices for the given band and hash.

        Parameters:
            band_id (int): The band identifier within the LSH structure.
            hash_val (bytes): The bucket hash value for which to retrieve indices.

        Returns:
            set: A set containing the indices (ints) of items stored in the specified bucket; empty if the bucket has no entries.
        """
        return set()

    def remove_indices(self, indices):
        """
        Remove the given item indices from storage.

        In this mock implementation the method is a no-op (indices are not persisted or removed).

        Parameters:
            indices (Iterable[int]): Iterable of integer indices to remove from storage.
        """
        pass

    def clear(self):
        """
        Clear mock storage state.

        This mock implementation performs no action; the method exists for API compatibility with the real storage used by tests.
        """
        pass


def test_concurrent_ingestion():
    """
    Test that multiple threads can ingest vectors concurrently without race conditions.
    """
    dim = 128
    num_vectors = 100
    num_threads = 10
    vectors_per_thread = num_vectors // num_threads

    # Setup LSH with mock storage
    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=10,
        rows_per_band=5,
        num_perm=50,
        storage=storage,
        buffer_size=10,  # Small buffer to force frequent flushes
    )

    def worker(thread_id: int):
        """
        Ingests a contiguous block of randomly generated vectors into the shared LSH instance.

        Generates `vectors_per_thread` random float32 vectors of dimension `dim` and calls `lsh.ingest` for each with sequential indices starting at `thread_id * vectors_per_thread`.

        Parameters:
            thread_id (int): Zero-based index of the worker thread; determines the starting vector index for this worker.
        """
        start_idx = thread_id * vectors_per_thread
        for i in range(vectors_per_thread):
            idx = start_idx + i
            vec = np.random.randn(dim).astype(np.float32)
            lsh.ingest(idx, vec)

    # Run threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Flush remaining items
    lsh.flush()

    # Verify total operations
    # Each vector produces num_bands signatures
    expected_ops = num_vectors * lsh._config["num_bands"]
    assert len(storage.operations) == expected_ops

    # Verify all indices are present
    indices = set()
    for _, _, idx in storage.operations:
        indices.add(idx)
    assert len(indices) == num_vectors
    assert max(indices) == num_vectors - 1


def test_flush_buffer_thread_safety():
    """
    Verifies that concurrently invoking flush causes each buffered item to be flushed exactly once across all bands.

    Populates the LSHRS buffer with items, starts multiple threads that call flush simultaneously, and asserts the storage received num_items × num_bands operations (ensuring no duplicate flushes).
    """
    dim = 64
    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=storage,
        buffer_size=1000,
    )

    # Add items to buffer without flushing
    num_items = 50
    for i in range(num_items):
        lsh.ingest(i, np.random.randn(dim))

    # Try to flush from multiple threads simultaneously
    def flusher():
        """
        Flush the shared LSHRS instance's buffered items to storage.

        Wrapper used by threads to invoke a flush operation on the module-scoped `lsh` instance.
        """
        lsh.flush()

    threads = [threading.Thread(target=flusher) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify operations were flushed exactly once (no duplicates)
    expected_ops = num_items * 4  # 4 bands
    assert len(storage.operations) == expected_ops
