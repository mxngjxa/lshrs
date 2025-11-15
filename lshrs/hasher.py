from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class HashSignatures:
    """Container for the hash signatures produced by a single vector."""
    bands: List[bytes]

    def __iter__(self) -> Iterable[bytes]:
        return iter(self.bands)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.bands)


class LSHHasher:
    """Random projection based LSH hasher."""

    def __init__(
        self,
        num_bands: int,
        rows_per_band: int,
        dim: int,
        *,
        seed: int = 42,
    ) -> None:
        if num_bands <= 0:
            raise ValueError("num_bands must be > 0")
        if rows_per_band <= 0:
            raise ValueError("rows_per_band must be > 0")
        if dim <= 0:
            raise ValueError("dim must be > 0")

        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.projections = [
            rng.standard_normal((rows_per_band, dim)).astype(np.float32)
            for _ in range(num_bands)
        ]

    def hash_vector(self, vector: np.ndarray) -> HashSignatures:
        """Hash a single vector into LSH band signatures."""
        vec = self._validate_vector(vector)
        bands = [
            self._project_and_pack(projection, vec)
            for projection in self.projections
        ]
        return HashSignatures(bands)

    def hash_batch(self, vectors: np.ndarray) -> List[HashSignatures]:
        """Hash a batch of vectors."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Batch input must be a 2D array")
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dimension {self.dim}, received {arr.shape[1]}"
            )
        return [self.hash_vector(vec) for vec in arr]

    def _project_and_pack(self, projection: np.ndarray, vector: np.ndarray) -> bytes:
        projected = projection @ vector
        binary = projected > 0
        packed = np.packbits(binary.astype(np.uint8), bitorder="little")
        return packed.tobytes()

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            raise ValueError(
                f"Expected vector of dimension {self.dim}, received {vec.shape}"
            )
        return vec