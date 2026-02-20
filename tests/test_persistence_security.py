"""Tests for LSHRS save/load persistence and pickle protocol."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from lshrs import LSHRS
from tests.conftest import MockStorage


def test_save_and_load_new_format(tmp_path):
    """Save and load preserves config and projection matrices exactly."""
    dim = 64
    storage = MockStorage()
    lsh = LSHRS(
        dim=dim,
        num_bands=8,
        rows_per_band=4,
        num_perm=32,
        storage=storage,
        seed=123,
    )

    save_path = tmp_path / "lsh_index"
    lsh.save_to_disk(save_path)

    assert save_path.is_dir()
    assert (save_path / "metadata.json").exists()
    assert (save_path / "projections.npz").exists()

    with open(save_path / "metadata.json") as f:
        metadata = json.load(f)
        assert metadata["config"]["dim"] == dim
        assert metadata["config"]["seed"] == 123
        assert "version" in metadata

    with np.load(save_path / "projections.npz") as data:
        assert len(data.files) == 8

    restored = LSHRS.load_from_disk(save_path, storage=MockStorage())

    assert restored._dim == lsh._dim
    assert restored._config == lsh._config

    for orig, rest in zip(lsh._hasher.projections, restored._hasher.projections, strict=True):
        np.testing.assert_array_equal(orig, rest)


def test_save_redacts_password(tmp_path):
    """Saved metadata must not contain the plaintext Redis password."""
    storage = MockStorage()
    lsh = LSHRS(
        dim=16,
        num_bands=2,
        rows_per_band=2,
        num_perm=4,
        storage=storage,
        redis_password="super_secret",
    )

    save_path = tmp_path / "redacted_test"
    lsh.save_to_disk(save_path)

    with open(save_path / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["redis_config"]["password"] == "<REDACTED>"


def test_load_missing_directory_raises(tmp_path):
    """Loading from a non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(tmp_path / "non_existent")


def test_load_missing_files_raises(tmp_path):
    """Loading from an incomplete directory raises appropriate errors."""
    save_path = tmp_path / "bad_index"
    save_path.mkdir()

    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(save_path)

    metadata = {
        "version": "0.1.1b2",
        "config": {
            "dim": 10,
            "num_perm": 10,
            "num_bands": 2,
            "rows_per_band": 5,
            "similarity_threshold": 0.5,
            "buffer_size": 100,
            "seed": 42,
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "prefix": "test",
            "decode_responses": False,
        },
    }
    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(save_path)


def test_pickle_round_trip():
    """Pickle serialization preserves config and projections."""
    storage = MockStorage()
    lsh = LSHRS(
        dim=32,
        num_bands=4,
        rows_per_band=4,
        num_perm=16,
        storage=storage,
        seed=7,
    )

    data = pickle.dumps(lsh)
    restored = pickle.loads(data)  # noqa: S301

    assert restored._config == lsh._config
    assert len(restored._hasher.projections) == len(lsh._hasher.projections)
    for orig, rest in zip(lsh._hasher.projections, restored._hasher.projections, strict=True):
        np.testing.assert_array_equal(orig, rest)
