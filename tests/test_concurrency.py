"""Tests for thread-safety of LSHRS ingestion and flush operations."""

from __future__ import annotations

import threading

import numpy as np

from lshrs import LSHRS
from tests.conftest import MockStorage


def test_concurrent_ingestion():
    """Multiple threads can ingest vectors without race conditions."""
    dim = 128
    num_vectors = 100
    num_threads = 10
    vectors_per_thread = num_vectors // num_threads
    rng = np.random.default_rng(42)

    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=10,
        rows_per_band=5,
        num_perm=50,
        storage=storage,
        buffer_size=10,
    )

    def worker(thread_id: int):
        start_idx = thread_id * vectors_per_thread
        for i in range(vectors_per_thread):
            idx = start_idx + i
            vec = rng.standard_normal(dim).astype(np.float32)
            lsh.ingest(idx, vec)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lsh.flush()

    expected_ops = num_vectors * lsh._config["num_bands"]
    assert storage.total_operations == expected_ops
    assert storage.unique_indices == set(range(num_vectors))


def test_flush_buffer_thread_safety():
    """Concurrent flush calls produce exactly the expected number of operations."""
    dim = 64
    rng = np.random.default_rng(99)
    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=storage,
        buffer_size=1000,
    )

    num_items = 50
    for i in range(num_items):
        lsh.ingest(i, rng.standard_normal(dim).astype(np.float32))

    threads = [threading.Thread(target=lsh.flush) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_ops = num_items * 4
    assert storage.total_operations == expected_ops
