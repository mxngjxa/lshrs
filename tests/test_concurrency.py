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
        self.batch_add_calls = 0
        self.operations = []
        self.lock = threading.Lock()

    def batch_add(self, operations: List[Any]) -> None:
        with self.lock:
            self.batch_add_calls += 1
            self.operations.extend(operations)
            # Simulate some IO latency
            time.sleep(0.001)
            
    def get_bucket(self, band_id: int, hash_val: bytes):
        return set()
        
    def remove_indices(self, indices):
        pass
        
    def clear(self):
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
        buffer_size=10  # Small buffer to force frequent flushes
    )
    
    def worker(thread_id: int):
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
    Test specifically that flush_buffer handles concurrent access correctly.
    """
    dim = 64
    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=storage,
        buffer_size=1000
    )
    
    # Add items to buffer without flushing
    num_items = 50
    for i in range(num_items):
        lsh.ingest(i, np.random.randn(dim))
        
    # Try to flush from multiple threads simultaneously
    def flusher():
        lsh.flush()
        
    threads = [threading.Thread(target=flusher) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # Verify operations were flushed exactly once (no duplicates)
    expected_ops = num_items * 4  # 4 bands
    assert len(storage.operations) == expected_ops