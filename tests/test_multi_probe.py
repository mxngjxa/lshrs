"""Tests for opt-in multi-probe LSH querying.

Covers:
    (a) probes=0 reproduces plain-LSH results exactly (regression).
    (b) probes>0 returns a SUPERSET of probes=0 candidates and can recover a
        near-neighbor that exact-bucket probing misses.
    (c) the bit-flip helper round-trips and is consistent with bitorder="little".
"""

from __future__ import annotations

import numpy as np

from lshrs.hash.lsh import LSHHasher

# ---------------------------------------------------------------------------
# (c) bit-flip helper
# ---------------------------------------------------------------------------


def test_flip_bit_roundtrips():
    """Flipping the same bit twice restores the original signature."""
    sig = bytes([0b1010_0101, 0b0000_1111])
    for bit in range(16):
        once = LSHHasher.flip_bit(sig, bit)
        assert once != sig
        twice = LSHHasher.flip_bit(once, bit)
        assert twice == sig


def test_flip_bit_matches_packbits_little_endian():
    """flip_bit must toggle the same logical bit np.packbits(bitorder='little') sets."""
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=16).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little").tobytes()

    for bit_index in range(16):
        flipped = LSHHasher.flip_bit(packed, bit_index)
        unpacked = np.unpackbits(np.frombuffer(flipped, dtype=np.uint8), bitorder="little")
        expected = bits.copy()
        expected[bit_index] ^= 1
        assert np.array_equal(unpacked, expected)


def test_hash_vector_with_margins_consistent_with_hash_vector():
    """Signatures from the margin variant match hash_vector; margin signs match bits."""
    hasher = LSHHasher(num_bands=3, rows_per_band=5, dim=16, seed=11)
    vec = np.random.default_rng(3).standard_normal(16)

    plain = hasher.hash_vector(vec)
    sigs, margins = hasher.hash_vector_with_margins(vec)

    assert sigs.as_tuple() == plain.as_tuple()
    assert len(margins) == 3
    for band_id, hash_val in enumerate(sigs):
        bits = np.unpackbits(np.frombuffer(hash_val, dtype=np.uint8), bitorder="little")
        bits = bits[: hasher.rows_per_band]
        assert np.array_equal(bits, (margins[band_id] > 0).astype(np.uint8))


# ---------------------------------------------------------------------------
# (a) regression: probes=0 == plain LSH
# ---------------------------------------------------------------------------


def test_probes_zero_matches_default(make_lsh, rng):
    """probes=0 returns exactly the same candidates as a default query."""
    lsh = make_lsh(dim=32, num_bands=8, rows_per_band=4, num_perm=32)
    vectors = rng.standard_normal((50, 32)).astype(np.float32)
    lsh.index(list(range(50)), vectors)

    query = rng.standard_normal(32).astype(np.float32)

    default = lsh.query(query, top_k=None)
    explicit_zero = lsh.query(query, top_k=None, probes=0)
    assert default == explicit_zero


# ---------------------------------------------------------------------------
# (b) probes>0 superset + recovers a missed near-neighbor
# ---------------------------------------------------------------------------


def test_probes_returns_superset(make_lsh, rng):
    """probes>0 candidate set must contain every probes=0 candidate."""
    lsh = make_lsh(dim=32, num_bands=8, rows_per_band=4, num_perm=32)
    vectors = rng.standard_normal((80, 32)).astype(np.float32)
    lsh.index(list(range(80)), vectors)

    query = rng.standard_normal(32).astype(np.float32)

    base = set(lsh.query(query, top_k=None, probes=0))
    probed = set(lsh.query(query, top_k=None, probes=4))
    assert base.issubset(probed)


def test_probes_recover_missed_near_neighbor(make_lsh):
    """Multi-probe recovers a near-neighbor that lands just over a single hyperplane.

    Construct a neighbor by taking the query and nudging it across the
    smallest-margin hyperplane of one band. That neighbor differs from the query
    by exactly one hash bit in that band, so exact-bucket probing for THAT band
    misses it, but a single-bit probe recovers it.
    """
    # Single band, so a 1-bit difference means zero exact collisions.
    lsh = make_lsh(dim=16, num_bands=1, rows_per_band=8, num_perm=8, seed=5)
    hasher = lsh._hasher

    query = np.random.default_rng(99).standard_normal(16).astype(np.float32)
    q_sig, q_margins = hasher.hash_vector_with_margins(query)

    # Find the smallest-|margin| bit in band 0 and build a neighbor whose only
    # hash difference is that single flipped bit.
    band_margins = q_margins[0]
    flip_bit = int(np.argmin(np.abs(band_margins)))

    # Search for a real vector that flips exactly that one bit and no others.
    neighbor = None
    rng = np.random.default_rng(1234)
    target_sig = q_sig[0]
    for _ in range(2000):
        cand = (query + rng.standard_normal(16).astype(np.float32) * 0.15).astype(np.float32)
        c_sig = hasher.hash_vector(cand)[0]
        if c_sig == LSHHasher.flip_bit(target_sig, flip_bit):
            neighbor = cand
            break
    assert neighbor is not None, "failed to construct a one-bit-off neighbor"

    lsh.index([0], neighbor.reshape(1, -1))

    # Exact-bucket probing misses it (different bucket), multi-probe finds it.
    assert lsh.query(query, top_k=None, probes=0) == []
    assert 0 in lsh.query(query, top_k=None, probes=1)
