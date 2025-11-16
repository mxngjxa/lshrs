import math

import numpy as np
import pytest

from lshrs._config.config import HashSignatures
from lshrs.hash.lsh import LSHHasher
from lshrs.utils.br import (
    compute_collision_probability,
    compute_false_rates,
    compute_lsh_threshold,
    get_optimal_config,
)
from lshrs.utils.norm import l2_norm
from lshrs.utils.similarity import cosine_similarity, top_k_cosine


@pytest.mark.parametrize(
    "num_bands, rows_per_band, dim",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ],
)
def test_lsh_hasher_invalid_init_parameters(num_bands, rows_per_band, dim):
    with pytest.raises(ValueError):
        LSHHasher(num_bands=num_bands, rows_per_band=rows_per_band, dim=dim)


def test_lsh_hash_vector_deterministic_and_shapes():
    num_bands = 3
    rows_per_band = 5
    dim = 4
    vector = np.arange(dim, dtype=np.float32)

    hasher_a = LSHHasher(
        num_bands=num_bands, rows_per_band=rows_per_band, dim=dim, seed=123
    )
    hasher_b = LSHHasher(
        num_bands=num_bands, rows_per_band=rows_per_band, dim=dim, seed=123
    )

    signatures_a = hasher_a.hash_vector(vector)
    signatures_b = hasher_b.hash_vector(vector)

    assert isinstance(signatures_a, HashSignatures)
    assert signatures_a.as_tuple() == signatures_b.as_tuple()
    assert len(signatures_a) == num_bands

    expected_band_length = math.ceil(rows_per_band / 8)

    for band in signatures_a:
        assert isinstance(band, bytes)
        assert len(band) == expected_band_length


def test_lsh_hash_vector_dimension_mismatch():
    hasher = LSHHasher(num_bands=2, rows_per_band=3, dim=4)

    with pytest.raises(ValueError):
        hasher.hash_vector(np.arange(5, dtype=np.float32))


def test_lsh_hash_batch_round_trip():
    hasher = LSHHasher(num_bands=4, rows_per_band=6, dim=3)

    vectors = np.array(
        [
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    signatures = hasher.hash_batch(vectors)

    assert len(signatures) == len(vectors)
    assert all(isinstance(sig, HashSignatures) for sig in signatures)
    assert signatures[0].as_tuple() != signatures[1].as_tuple()


def test_lsh_hash_batch_validates_input():
    hasher = LSHHasher(num_bands=2, rows_per_band=4, dim=3)

    with pytest.raises(ValueError):
        hasher.hash_batch(np.arange(3, dtype=np.float32))

    mismatched = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        hasher.hash_batch(mismatched)


def test_hash_signatures_normalizes_iterables():
    signatures = HashSignatures((b"\x00\x01", bytearray(b"\x02\x03")))

    assert signatures.as_tuple() == (b"\x00\x01", b"\x02\x03")
    assert list(signatures) == [b"\x00\x01", b"\x02\x03"]
    assert len(signatures) == 2


def test_l2_norm_returns_unit_vector():
    vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)

    normalized = l2_norm(vector)

    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    assert normalized.shape == (3,)


def test_l2_norm_zero_vector_raises():
    with pytest.raises(ValueError):
        l2_norm(np.zeros(4, dtype=np.float32))


def test_cosine_similarity_expected_values():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    ]

    similarities = cosine_similarity(query, candidates)

    expected = np.array(
        [1.0, 0.0, -1.0, 0.70710677],
        dtype=np.float32,
    )

    assert similarities.shape == (4,)
    assert np.allclose(similarities, expected, atol=1e-6)


def test_top_k_cosine_returns_sorted_indices():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = [
        np.array([1.0, 0.1, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.2, 0.0], dtype=np.float32),
    ]

    top_results = top_k_cosine(query, candidates, k=3)

    assert [idx for idx, _ in top_results] == [2, 0, 4]
    assert top_results[0][1] == pytest.approx(1.0)
    assert top_results[1][1] >= top_results[2][1]

    all_results = top_k_cosine(query, candidates, k=10)

    assert len(all_results) == len(candidates)


def test_top_k_cosine_invalid_k():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = [np.array([1.0, 0.0], dtype=np.float32)]

    with pytest.raises(ValueError):
        top_k_cosine(query, candidates, k=0)


def test_compute_lsh_threshold_matches_closed_form():
    b, r = 64, 8

    assert compute_lsh_threshold(b, r) == pytest.approx((1 / b) ** (1 / r))


def test_compute_collision_probability_monotonic():
    b, r = 32, 6

    low_similarity = compute_collision_probability(0.2, b, r)
    high_similarity = compute_collision_probability(0.8, b, r)

    assert 0.0 <= low_similarity < high_similarity <= 1.0


def test_compute_false_rates_bounds():
    b, r, threshold = 4, 4, 0.7

    fp_rate, fn_rate = compute_false_rates(b, r, threshold)

    assert 0.0 <= fp_rate <= 1.0
    assert 0.0 <= fn_rate <= 1.0


def test_get_optimal_config_prefers_precomputed_entries():
    b, r = get_optimal_config(4096, 0.9)

    assert (b, r) == (64, 64)

    actual_threshold = compute_lsh_threshold(b, r)

    assert actual_threshold == pytest.approx((1 / 64) ** (1 / 64))
    assert abs(actual_threshold - 0.9) < 0.05
