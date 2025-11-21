import json
# import shutil
# from pathlib import Path

import numpy as np
import pytest

from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage


class MockStorage(RedisStorage):
    def __init__(self):
        pass
    def batch_add(self, ops):
        pass
    def clear(self):
        pass


def test_save_and_load_new_format(tmp_path):
    """
    Test saving and loading LSHRS instance using the new secure JSON/NumPy format.
    """
    dim = 64
    lsh = LSHRS(
        dim=dim,
        num_bands=8,
        rows_per_band=4,
        num_perm=32,
        storage=MockStorage(),
        seed=123
    )
    
    save_path = tmp_path / "lsh_index"
    lsh.save_to_disk(save_path)
    
    # Verify directory structure
    assert save_path.is_dir()
    assert (save_path / "metadata.json").exists()
    assert (save_path / "projections.npz").exists()
    
    # Verify metadata is valid JSON
    with open(save_path / "metadata.json", "r") as f:
        metadata = json.load(f)
        assert metadata["config"]["dim"] == dim
        assert metadata["config"]["seed"] == 123
        assert "version" in metadata
        
    # Verify projections are valid NPZ
    with np.load(save_path / "projections.npz") as data:
        assert len(data.files) == 8  # num_bands
        
    # Load back
    restored = LSHRS.load_from_disk(save_path, storage=MockStorage())
    
    # Verify restored state matches original
    assert restored._dim == lsh._dim
    assert restored._config == lsh._config
    
    # Verify projections match exactly
    for orig, rest in zip(lsh._hasher.projections, restored._hasher.projections):
        np.testing.assert_array_equal(orig, rest)


def test_load_missing_directory_raises(tmp_path):
    """Test that loading from a non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(tmp_path / "non_existent")


def test_load_missing_files_raises(tmp_path):
    """Test that loading from a directory missing required files raises error."""
    save_path = tmp_path / "bad_index"
    save_path.mkdir()
    
    # Missing metadata.json
    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(save_path)
        
    # Create valid metadata but missing projections
    metadata = {
        "version": "0.1.1a4",
        "config": {
            "dim": 10, "num_perm": 10, "num_bands": 2, "rows_per_band": 5,
            "similarity_threshold": 0.5, "buffer_size": 100, "seed": 42
        },
        "redis_config": {
            "host": "localhost", "port": 6379, "db": 0,
            "password": None, "prefix": "test", "decode_responses": False
        }
    }
    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
        
    with pytest.raises(FileNotFoundError):
        LSHRS.load_from_disk(save_path)