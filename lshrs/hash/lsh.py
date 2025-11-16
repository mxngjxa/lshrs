"""
Locality-Sensitive Hashing (LSH) Implementation using Random Projections

This module implements LSH for approximate nearest neighbor search in high-dimensional
spaces. It uses random hyperplane projections to hash similar vectors to the same buckets.

The banding technique divides hash signatures into multiple bands, allowing control over
the similarity threshold for candidate matches.
"""

from __future__ import annotations

from typing import List

import numpy as np

from lshrs._config.config import HashSignatures


class LSHHasher:
    """
    Random projection based Locality-Sensitive Hashing (LSH) hasher.

    This hasher uses random hyperplane projections to convert high-dimensional vectors
    into compact binary signatures. The algorithm:

    1. Projects each vector onto random hyperplanes (one set per band)
    2. Creates a binary hash: 1 if projection > 0, else 0
    3. Packs bits into bytes for efficient storage and comparison

    The banding technique (num_bands × rows_per_band) controls the trade-off between
    recall (finding all similar items) and precision (avoiding false positives).

    Theory:
        - More bands → higher recall (find more similar pairs)
        - More rows per band → higher precision (fewer false positives)
        - Total hash bits = num_bands × rows_per_band

    Typical usage:
        >>> hasher = LSHHasher(num_bands=20, rows_per_band=5, dim=128)
        >>> vector = np.random.randn(128).astype(np.float32)
        >>> signatures = hasher.hash_vector(vector)
        >>> # Store signatures[0], signatures[1], etc. in separate hash tables

    Attributes:
        num_bands: Number of independent hash bands (separate hash tables)
        rows_per_band: Number of hash bits per band (hyperplane projections)
        dim: Expected dimensionality of input vectors
        projections: List of random projection matrices, one per band
                     Each matrix has shape (rows_per_band, dim)
    """

    def __init__(
        self,
        num_bands: int,
        rows_per_band: int,
        dim: int,
        seed: int = 42,
    ) -> None:
        """
        Initialize the LSH hasher with random projection matrices.

        Args:
            num_bands: Number of hash bands (independent hash tables).
                      More bands → better recall but more storage.
            rows_per_band: Number of hyperplane projections per band.
                          More rows → better precision but longer signatures.
            dim: Dimensionality of input vectors (must match during hashing).
            seed: Random seed for reproducible projection matrices.
                 Use different seeds for different hasher instances.

        Raises:
            ValueError: If any parameter is <= 0.

        Example:
            >>> # For 128-dim vectors, use 20 bands × 5 rows = 100 hash bits total
            >>> hasher = LSHHasher(num_bands=20, rows_per_band=5, dim=128, seed=42)
        """
        # Validate parameters
        if num_bands <= 0:
            raise ValueError("num_bands must be > 0")
        if rows_per_band <= 0:
            raise ValueError("rows_per_band must be > 0")
        if dim <= 0:
            raise ValueError("dim must be > 0")

        # Store configuration
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.dim = dim

        # Generate random projection matrices (one per band)
        # Each matrix projects dim-dimensional vectors to rows_per_band dimensions
        # Using standard normal distribution (mean=0, std=1) is theoretically sound
        rng = np.random.default_rng(seed)
        self.projections = [
            rng.standard_normal((rows_per_band, dim)).astype(np.float32)
            for _ in range(num_bands)
        ]

    def hash_vector(self, vector: np.ndarray) -> HashSignatures:
        """
        Hash a single vector into LSH band signatures.

        Process:
            1. Validate vector shape and convert to float32
            2. For each band:
               a. Project vector using random hyperplanes
               b. Convert projections to binary (>0 → 1, ≤0 → 0)
               c. Pack bits into bytes for compact storage
            3. Return all band signatures wrapped in HashSignatures

        Args:
            vector: Input vector to hash. Can be any array-like of length `dim`.
                   Will be reshaped to 1D if needed.

        Returns:
            HashSignatures containing one binary signature per band.
            Each signature is a bytes object of length ceil(rows_per_band / 8).

        Raises:
            ValueError: If vector dimension doesn't match expected `dim`.

        Example:
            >>> hasher = LSHHasher(num_bands=3, rows_per_band=8, dim=64)
            >>> vec = np.random.randn(64)
            >>> sigs = hasher.hash_vector(vec)
            >>> len(sigs)  # 3 bands
            3
            >>> len(sigs.bands[0])  # 1 byte per band (8 bits packed)
            1
        """
        # Validate and normalize input vector
        vec = self._validate_vector(vector)

        # Hash vector with each band's projection matrix
        bands = [
            self._project_and_pack(projection, vec) for projection in self.projections
        ]

        return HashSignatures(bands)

    def hash_batch(self, vectors: np.ndarray) -> List[HashSignatures]:
        """
        Hash a batch of vectors efficiently.

        This method validates the entire batch upfront, then hashes each vector
        individually. For very large batches, consider processing in chunks to
        manage memory usage.

        Args:
            vectors: 2D array of shape (num_vectors, dim).
                    Each row is a vector to hash.

        Returns:
            List of HashSignatures, one per input vector, in the same order.

        Raises:
            ValueError: If input is not 2D or vectors don't match expected `dim`.

        Example:
            >>> hasher = LSHHasher(num_bands=10, rows_per_band=4, dim=128)
            >>> batch = np.random.randn(100, 128)  # 100 vectors
            >>> signatures = hasher.hash_batch(batch)
            >>> len(signatures)  # 100 HashSignatures objects
            100
        """
        # Convert to float32 array and validate shape
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Batch input must be a 2D array")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dimension {self.dim}, received {arr.shape[1]}"
            )

        # Hash each vector in the batch
        return [self.hash_vector(vec) for vec in arr]

    def _project_and_pack(self, projection: np.ndarray, vector: np.ndarray) -> bytes:
        """
        Project vector onto random hyperplanes and pack result into bytes.

        This is the core LSH operation for a single band:
            1. Matrix multiply: projection @ vector gives signed real values
            2. Threshold at zero: positive values → 1 bit, negative/zero → 0 bit
            3. Pack bits into bytes using little-endian bit order

        Args:
            projection: Random projection matrix of shape (rows_per_band, dim).
            vector: Input vector of shape (dim,).

        Returns:
            Packed binary signature as bytes. Length is ceil(rows_per_band / 8).

        Technical notes:
            - Uses little-endian bit packing for consistency
            - Resulting bytes can be used as dictionary keys or stored in databases
            - Two vectors with similar directions will likely produce identical signatures

        Example:
            >>> projection = np.random.randn(4, 10).astype(np.float32)
            >>> vector = np.random.randn(10).astype(np.float32)
            >>> sig = hasher._project_and_pack(projection, vector)
            >>> len(sig)  # 4 bits packed into 1 byte
            1
        """
        # Project vector onto random hyperplanes (matrix-vector multiply)
        projected = projection @ vector

        # Convert to binary: 1 if positive, 0 if negative/zero
        # This creates a hyperplane-based hash: which side of each plane?
        binary = projected > 0

        # Pack boolean array into compact byte representation
        # bitorder='little' ensures consistent packing across platforms
        packed = np.packbits(binary.astype(np.uint8), bitorder="little")

        # Convert numpy array to Python bytes for hashing/storage
        return packed.tobytes()

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input vector to expected shape and dtype.

        Ensures the vector:
            - Is converted to float32 (memory efficient and sufficient precision)
            - Is 1D with exactly `dim` elements
            - Raises clear error messages for invalid inputs

        Args:
            vector: Input vector (can be list, tuple, or any array-like).

        Returns:
            Validated vector as 1D float32 numpy array of shape (dim,).

        Raises:
            ValueError: If vector dimension doesn't match expected `dim`.

        Example:
            >>> hasher = LSHHasher(num_bands=5, rows_per_band=4, dim=128)
            >>> vec = [1, 2, 3, ..., 128]  # list input
            >>> validated = hasher._validate_vector(vec)
            >>> validated.shape
            (128,)
            >>> validated.dtype
            dtype('float32')
        """
        # Convert to float32 and flatten to 1D
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)

        # Check dimension matches expected
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dimension {self.dim}, received {vec.shape}"
            )

        return vec
