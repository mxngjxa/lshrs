#!/usr/bin/env python3
"""
Migration script to convert legacy pickle-based LSH indices to the new secure JSON/NumPy format.

Usage:
    python bin/migrate_to_json.py <input_pickle_file> <output_directory>

Example:
    python bin/migrate_to_json.py old_index.pkl new_index_dir
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Ensure lshrs is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lshrs import LSHRS


def migrate(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    output_dir = Path(output_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Error: Output directory '{output_dir}' already exists and is not empty.")
        sys.exit(1)

    print(f"Loading legacy index from '{input_file}'...")
    
    try:
        with open(input_file, "rb") as f:
            state = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    # Verify it looks like a legacy LSHRS dump
    required_keys = {"config", "redis_config", "projections"}
    if not isinstance(state, dict) or not required_keys.issubset(state.keys()):
        print("Error: Input file does not appear to be a valid legacy LSHRS index.")
        sys.exit(1)

    config = state["config"]
    redis_config = state["redis_config"]
    projections = state["projections"]

    print(f"  - Dimension: {config['dim']}")
    print(f"  - Bands: {config['num_bands']}")
    print(f"  - Rows per band: {config['rows_per_band']}")

    # Reconstruct instance
    # We pass None for storage to avoid connecting to Redis during migration
    # (unless strictly necessary, but save_to_disk doesn't need Redis)
    # However, LSHRS init connects to Redis by default.
    # We'll let it connect or fail if Redis isn't there?
    # Ideally migration shouldn't require Redis.
    # But LSHRS.__init__ creates RedisStorage which makes a connection pool.
    # It doesn't necessarily connect immediately? 
    # Actually, let's try to mock storage or allow lazy connection?
    # For now, we assume we can instantiate it.
    
    print("Reconstructing LSHRS instance...")
    try:
        lsh = LSHRS(
            dim=config["dim"],
            num_perm=config["num_perm"],
            num_bands=config["num_bands"],
            rows_per_band=config["rows_per_band"],
            similarity_threshold=config["similarity_threshold"],
            buffer_size=config["buffer_size"],
            redis_host=redis_config["host"],
            redis_port=redis_config["port"],
            redis_db=redis_config["db"],
            redis_password=redis_config["password"],
            redis_prefix=redis_config["prefix"],
            decode_responses=redis_config["decode_responses"],
            seed=config["seed"],
        )
    except Exception as e:
        print(f"Warning: Failed to connect to Redis ({e}). Creating instance with dummy storage for migration.")
        # Mock storage to allow saving without Redis connection
        from lshrs.storage.redis import RedisStorage
        class DummyStorage(RedisStorage):
            def __init__(self): pass
            def batch_add(self, ops): pass
        
        lsh = LSHRS(
            dim=config["dim"],
            num_perm=config["num_perm"],
            num_bands=config["num_bands"],
            rows_per_band=config["rows_per_band"],
            similarity_threshold=config["similarity_threshold"],
            buffer_size=config["buffer_size"],
            storage=DummyStorage(),
            seed=config["seed"],
        )
        # Restore redis config for metadata
        lsh._redis_config = redis_config

    # Inject saved projections
    lsh._hasher.projections = [
        np.asarray(matrix, dtype=np.float32) for matrix in projections
    ]

    print(f"Saving to new format at '{output_dir}'...")
    lsh.save_to_disk(output_dir)
    
    print("Migration successful!")
    print(f"You can now load the index using: LSHRS.load_from_disk('{output_dir}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate LSHRS pickle index to JSON/NumPy format.")
    parser.add_argument("input", help="Path to legacy .pkl file")
    parser.add_argument("output", help="Path to output directory")
    
    args = parser.parse_args()
    migrate(args.input, args.output)