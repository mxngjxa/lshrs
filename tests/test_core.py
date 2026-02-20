"""Comprehensive tests for LSHRS core: constructor, query, delete, stats, error paths."""

from __future__ import annotations

import numpy as np
import pytest

from lshrs import LSHRS
from tests.conftest import MockStorage

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="dimensionality must be greater than zero"):
            LSHRS(dim=0, num_bands=2, rows_per_band=2, num_perm=4, storage=MockStorage())

    def test_num_perm_must_be_positive(self):
        with pytest.raises(ValueError, match="num_perm must be greater than zero"):
            LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=0, storage=MockStorage())

    def test_buffer_size_must_be_positive(self):
        with pytest.raises(ValueError, match="buffer_size must be greater than zero"):
            LSHRS(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=0, storage=MockStorage())

    def test_bands_times_rows_must_equal_num_perm(self):
        with pytest.raises(ValueError, match="num_bands .* rows_per_band must equal num_perm"):
            LSHRS(dim=4, num_bands=3, rows_per_band=3, num_perm=10, storage=MockStorage())

    def test_auto_config_when_bands_not_specified(self):
        lsh = LSHRS(dim=4, num_perm=128, storage=MockStorage())
        assert lsh._config["num_bands"] * lsh._config["rows_per_band"] == 128


# ---------------------------------------------------------------------------
# Ingest validation
# ---------------------------------------------------------------------------


class TestIngestValidation:
    def test_negative_index_rejected(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(ValueError, match="non-negative"):
            lsh.ingest(-1, np.ones(4, dtype=np.float32))

    def test_dimension_mismatch_rejected(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(ValueError, match="dimension"):
            lsh.ingest(0, np.ones(8, dtype=np.float32))

    def test_zero_vector_rejected(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(ValueError, match="zero vector"):
            lsh.ingest(0, np.zeros(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# Index (batch) validation
# ---------------------------------------------------------------------------


class TestBatchIndex:
    def test_empty_indices_is_noop(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        lsh.index([], np.zeros((0, 4), dtype=np.float32))
        assert mock_storage.total_operations == 0

    def test_shape_mismatch_raises(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(ValueError, match="shape"):
            lsh.index([0, 1], np.ones((2, 8), dtype=np.float32))

    def test_count_mismatch_raises(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(ValueError, match="does not match"):
            lsh.index([0, 1, 2], np.ones((2, 4), dtype=np.float32))

    def test_index_with_vector_fetch_fn(self, mock_storage):
        """index() without vectors should use vector_fetch_fn."""
        dim = 8
        vecs = np.eye(dim, dtype=np.float32)[:3]

        def fetch_fn(indices):
            return vecs

        lsh = LSHRS(
            dim=dim,
            num_bands=2,
            rows_per_band=2,
            num_perm=4,
            storage=mock_storage,
            vector_fetch_fn=fetch_fn,
        )
        lsh.index([0, 1, 2])
        assert mock_storage.total_operations == 3 * 2  # 3 vectors * 2 bands

    def test_index_without_vectors_or_fetch_fn_raises(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        with pytest.raises(RuntimeError, match="vector_fetch_fn"):
            lsh.index([0, 1])


# ---------------------------------------------------------------------------
# End-to-end query accuracy
# ---------------------------------------------------------------------------


class TestQueryAccuracy:
    def test_identical_vector_found(self, mock_storage, rng):
        """A vector should be its own best match."""
        dim = 32
        lsh = LSHRS(dim=dim, num_bands=8, rows_per_band=4, num_perm=32, storage=mock_storage, seed=42)

        target = rng.standard_normal(dim).astype(np.float32)
        # Index the target and some distractors
        lsh.ingest(0, target)
        for i in range(1, 20):
            lsh.ingest(i, rng.standard_normal(dim).astype(np.float32))
        lsh.flush()

        results = lsh.query(target, top_k=5)
        assert 0 in results

    def test_similar_vectors_ranked_above_dissimilar(self, mock_storage, rng):
        """Vectors close to the query should appear before random ones."""
        dim = 64
        lsh = LSHRS(dim=dim, num_bands=16, rows_per_band=4, num_perm=64, storage=mock_storage, seed=42)

        base = rng.standard_normal(dim).astype(np.float32)
        # Index a near-duplicate (small perturbation)
        near = base + rng.standard_normal(dim).astype(np.float32) * 0.01
        lsh.ingest(0, near)

        # Index distant vectors
        for i in range(1, 30):
            lsh.ingest(i, rng.standard_normal(dim).astype(np.float32))
        lsh.flush()

        results = lsh.query(base, top_k=5)
        # The near-duplicate should be among top results
        assert 0 in results

    def test_query_returns_empty_when_no_data(self, make_lsh, rng):
        """Querying with no indexed data returns empty list."""
        lsh = make_lsh(dim=32)
        results = lsh.query(rng.standard_normal(32).astype(np.float32), top_k=10)
        assert results == []


# ---------------------------------------------------------------------------
# Query validation and modes
# ---------------------------------------------------------------------------


class TestQueryValidation:
    def test_top_k_zero_raises(self, mock_storage, make_lsh, rng):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            lsh.query(vec, top_k=0)

    def test_top_k_negative_raises(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            lsh.query(vec, top_k=-1)

    def test_top_p_out_of_range_raises(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()
        with pytest.raises(ValueError, match="top_p must be within"):
            lsh.query(vec, top_p=0.0)
        with pytest.raises(ValueError, match="top_p must be within"):
            lsh.query(vec, top_p=1.5)

    def test_top_k_none_returns_all(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.ingest(1, np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32))
        lsh.flush()
        results = lsh.query(vec, top_k=None)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Top-p / reranking queries
# ---------------------------------------------------------------------------


class TestTopPQuery:
    def test_top_p_returns_scores(self, mock_storage):
        dim = 8
        vecs = np.eye(dim, dtype=np.float32)

        def fetch_fn(indices):
            return np.array([vecs[i] for i in indices], dtype=np.float32)

        lsh = LSHRS(
            dim=dim,
            num_bands=2,
            rows_per_band=2,
            num_perm=4,
            storage=mock_storage,
            vector_fetch_fn=fetch_fn,
        )

        for i in range(dim):
            lsh.ingest(i, vecs[i])
        lsh.flush()

        results = lsh.query(vecs[0], top_p=1.0, top_k=None)
        # Results should be list of (index, similarity) tuples
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            idx, score = item
            assert isinstance(score, float)

    def test_top_p_without_fetch_fn_raises(self, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()
        with pytest.raises(RuntimeError, match="vector_fetch_fn"):
            lsh.query(vec, top_p=0.5)

    def test_get_above_p_wrapper(self, mock_storage):
        dim = 4
        vecs = np.eye(dim, dtype=np.float32)

        def fetch_fn(indices):
            return np.array([vecs[i] for i in indices], dtype=np.float32)

        lsh = LSHRS(
            dim=dim,
            num_bands=2,
            rows_per_band=2,
            num_perm=4,
            storage=mock_storage,
            vector_fetch_fn=fetch_fn,
        )

        for i in range(dim):
            lsh.ingest(i, vecs[i])
        lsh.flush()

        results = lsh.get_above_p(vecs[0], p=1.0)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_get_top_k_wrapper(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()

        results = lsh.get_top_k(vec, topk=5)
        assert isinstance(results, list)
        assert 0 in results


# ---------------------------------------------------------------------------
# Delete, clear, stats
# ---------------------------------------------------------------------------


class TestMaintenanceOps:
    def test_delete_single_index(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        lsh.delete(42)
        assert mock_storage.removed_indices == [[42]]

    def test_delete_multiple_indices(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        lsh.delete([10, 20, 30])
        assert mock_storage.removed_indices == [[10, 20, 30]]

    def test_delete_removes_from_buckets(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        lsh.ingest(0, vec)
        lsh.flush()

        # Confirm it's found
        results = lsh.query(vec, top_k=1)
        assert results == [0]

        # Delete and confirm it's gone
        lsh.delete(0)
        results = lsh.query(vec, top_k=1)
        assert results == []

    def test_clear_delegates_to_storage(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4)
        lsh.ingest(0, np.ones(4, dtype=np.float32))
        lsh.clear()
        assert mock_storage.clear_called

    def test_clear_flushes_buffer_first(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=4, num_bands=2, rows_per_band=2, num_perm=4, buffer_size=1000)
        lsh.ingest(0, np.ones(4, dtype=np.float32))
        assert mock_storage.batch_add_call_count == 0
        lsh.clear()
        # Buffer should have been flushed before clear
        assert mock_storage.batch_add_call_count == 1
        assert mock_storage.clear_called

    def test_stats_returns_correct_config(self, mock_storage, make_lsh):
        lsh = make_lsh(dim=32, num_bands=4, rows_per_band=4, num_perm=16)
        info = lsh.stats()

        assert info["dimension"] == 32
        assert info["num_bands"] == 4
        assert info["rows_per_band"] == 4
        assert info["num_perm"] == 16
        assert "buffer_size" in info
        assert "similarity_threshold" in info
        assert "redis_prefix" in info


# ---------------------------------------------------------------------------
# Flush error recovery
# ---------------------------------------------------------------------------


class TestFlushErrorRecovery:
    def test_buffer_restored_on_flush_failure(self):
        """If batch_add fails, operations should be restored to the buffer."""
        storage = MockStorage(fail_on_flush=True)
        lsh = LSHRS(
            dim=4,
            num_bands=2,
            rows_per_band=2,
            num_perm=4,
            storage=storage,
            buffer_size=1000,
        )

        lsh.ingest(0, np.ones(4, dtype=np.float32))
        assert len(lsh._buffer) == 2  # 2 bands

        with pytest.raises(ConnectionError):
            lsh.flush()

        # Operations should be restored to the buffer
        assert len(lsh._buffer) == 2


# ---------------------------------------------------------------------------
# Resolve loader
# ---------------------------------------------------------------------------


class TestResolveLoader:
    def test_invalid_format_raises(self, make_lsh):
        lsh = make_lsh()
        with pytest.raises(ValueError, match="Unsupported"):
            lsh.create_signatures(format="csv")

    def test_postgres_format_aliases(self, make_lsh):
        lsh = make_lsh()
        # Should not raise (just resolve the loader)
        loader = lsh._resolve_loader("postgres")
        assert callable(loader)
        loader = lsh._resolve_loader("pg")
        assert callable(loader)

    def test_parquet_format_aliases(self, make_lsh):
        lsh = make_lsh()
        loader = lsh._resolve_loader("parquet")
        assert callable(loader)
        loader = lsh._resolve_loader("pq")
        assert callable(loader)


# ---------------------------------------------------------------------------
# Different seeds produce different hashes
# ---------------------------------------------------------------------------


class TestSeedBehavior:
    def test_same_seed_same_hashes(self, mock_storage):
        dim = 16
        vec = np.ones(dim, dtype=np.float32)

        lsh_a = LSHRS(dim=dim, num_bands=2, rows_per_band=2, num_perm=4, storage=MockStorage(), seed=42)
        lsh_b = LSHRS(dim=dim, num_bands=2, rows_per_band=2, num_perm=4, storage=MockStorage(), seed=42)

        sig_a = lsh_a._hasher.hash_vector(vec)
        sig_b = lsh_b._hasher.hash_vector(vec)
        assert sig_a.as_tuple() == sig_b.as_tuple()

    def test_different_seeds_different_hashes(self, mock_storage):
        dim = 64
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(dim).astype(np.float32)

        lsh_a = LSHRS(dim=dim, num_bands=8, rows_per_band=8, num_perm=64, storage=MockStorage(), seed=1)
        lsh_b = LSHRS(dim=dim, num_bands=8, rows_per_band=8, num_perm=64, storage=MockStorage(), seed=999)

        sig_a = lsh_a._hasher.hash_vector(vec)
        sig_b = lsh_b._hasher.hash_vector(vec)
        assert sig_a.as_tuple() != sig_b.as_tuple()
