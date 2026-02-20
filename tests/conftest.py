"""Shared test fixtures and mock implementations."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from lshrs import LSHRS
from lshrs.storage.redis import BucketOperation, RedisStorage


class MockStorage(RedisStorage):
    """Thread-safe in-memory mock that records operations and supports real bucket queries."""

    def __init__(self, *, fail_on_flush: bool = False) -> None:
        # Bucket data: (band_id, hash_hex) -> set of indices
        self.data: dict[tuple[int, str], set[int]] = {}
        # Record every batch_add call for assertion
        self.batches: list[list[BucketOperation]] = []
        self.all_operations: list[BucketOperation] = []
        self.batch_add_call_count: int = 0
        # Tracking flags
        self.close_called: bool = False
        self.clear_called: bool = False
        self.removed_indices: list[list[int]] = []
        # Thread safety
        self._lock = threading.Lock()
        # Optional failure simulation
        self._fail_on_flush = fail_on_flush

    def batch_add(self, operations: list[Any]) -> None:
        if self._fail_on_flush:
            raise ConnectionError("Simulated Redis failure")
        with self._lock:
            self.batch_add_call_count += 1
            self.batches.append(list(operations))
            self.all_operations.extend(operations)
            for band_id, hash_val, index in operations:
                key = (band_id, hash_val.hex() if isinstance(hash_val, bytes) else str(hash_val))
                self.data.setdefault(key, set()).add(index)

    def get_bucket(self, band_id: int, hash_val: bytes) -> set[int]:
        key = (band_id, hash_val.hex())
        with self._lock:
            return set(self.data.get(key, set()))

    def add_to_bucket(self, band_id: int, hash_val: bytes, index: int) -> None:
        key = (band_id, hash_val.hex())
        with self._lock:
            self.data.setdefault(key, set()).add(index)

    def remove_indices(self, indices: list[int]) -> None:
        with self._lock:
            self.removed_indices.append(list(indices))
            idx_set = set(indices)
            for key in self.data:
                self.data[key] -= idx_set

    def clear(self) -> None:
        with self._lock:
            self.clear_called = True
            self.data.clear()

    def close(self) -> None:
        self.close_called = True

    @property
    def total_operations(self) -> int:
        with self._lock:
            return len(self.all_operations)

    @property
    def unique_indices(self) -> set[int]:
        with self._lock:
            return {idx for _, _, idx in self.all_operations}


@pytest.fixture
def mock_storage() -> MockStorage:
    """Fresh MockStorage instance."""
    return MockStorage()


@pytest.fixture
def make_lsh(mock_storage: MockStorage):
    """Factory for creating LSHRS with MockStorage and sensible test defaults."""

    def _make(
        dim: int = 32,
        num_bands: int = 4,
        rows_per_band: int = 4,
        num_perm: int = 16,
        buffer_size: int = 10_000,
        seed: int = 42,
        vector_fetch_fn=None,
        storage=None,
    ) -> LSHRS:
        return LSHRS(
            dim=dim,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            num_perm=num_perm,
            buffer_size=buffer_size,
            seed=seed,
            vector_fetch_fn=vector_fetch_fn,
            storage=storage or mock_storage,
        )

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for deterministic tests."""
    return np.random.default_rng(12345)
