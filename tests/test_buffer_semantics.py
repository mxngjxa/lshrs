"""Tests for LSHRS buffer management: flush timing, auto-flush, and close behavior."""

from __future__ import annotations

import numpy as np

from tests.conftest import MockStorage


def test_single_ingest_not_immediately_queryable(mock_storage: MockStorage, make_lsh):
    """Single ingestion is buffered and not immediately searchable."""
    lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    lsh.ingest(0, vec)
    assert len(mock_storage.batches) == 0

    # Query should return empty because storage is empty
    results = lsh.query(vec, top_k=1)
    assert results == []

    lsh.flush()
    assert len(mock_storage.batches) == 1

    # Now query should find the vector
    results = lsh.query(vec, top_k=1)
    assert results == [0]


def test_batch_index_auto_flushes(mock_storage: MockStorage, make_lsh):
    """batch index() forces a flush at the end."""
    lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100)

    vecs = np.eye(4, dtype=np.float32)
    indices = [0, 1, 2, 3]

    lsh.index(indices, vecs)

    assert len(mock_storage.batches) >= 1
    total_ops = sum(len(b) for b in mock_storage.batches)
    # 4 vectors * 2 bands = 8 operations
    assert total_ops == 8

    # Immediately queryable after index()
    results = lsh.query(vecs[0], top_k=1)
    assert results == [0]


def test_buffer_flush_on_full(mock_storage: MockStorage, make_lsh):
    """Buffer auto-flushes when capacity is reached."""
    # buffer_size=4. Each vector adds 2 ops (2 bands). 2 vectors = 4 ops -> flush
    lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=4)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    lsh.ingest(0, vec)
    assert len(mock_storage.batches) == 0

    lsh.ingest(1, vec)
    assert len(mock_storage.batches) == 1
    assert len(mock_storage.batches[0]) == 4


def test_close_flushes_buffer(mock_storage: MockStorage, make_lsh):
    """Closing the instance flushes pending operations."""
    lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=100)

    lsh.ingest(0, np.ones(4, dtype=np.float32))
    assert len(mock_storage.batches) == 0

    lsh.close()
    assert len(mock_storage.batches) == 1
    assert mock_storage.close_called


def test_flush_empty_buffer_is_noop(mock_storage: MockStorage, make_lsh):
    """Flushing an empty buffer should do nothing."""
    lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
    lsh.flush()
    assert mock_storage.batch_add_call_count == 0
