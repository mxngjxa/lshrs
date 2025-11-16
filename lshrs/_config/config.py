"""
The config module holds package-wide configurables and provides
a uniform API for working with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

@dataclass(frozen=True)
class HashSignatures:
    """
    Container for LSH hash signatures produced by a single vector.

    Each vector is hashed into multiple 'bands', where each band contains a binary
    signature. Vectors with matching signatures in ANY band are considered candidate
    similar pairs. The banding technique allows tuning the similarity threshold.

    Attributes:
        bands: List of byte strings, one per band. Each byte string represents
               a packed binary hash signature (the result of multiple random projections).

    Example:
        >>> sigs = HashSignatures([b'\x01\x02', b'\xff\x00', b'\xaa\xbb'])
        >>> len(sigs)  # Number of bands
        3
        >>> for band_sig in sigs:
        ...     print(band_sig.hex())  # Print each band's signature in hex
    """

    bands: Set[bytes]

    def __iter__(self) -> Iterable[bytes]:
        """
        Iterate over band signatures.

        Returns:
            Iterator yielding each band's binary signature.
        """
        return iter(self.bands)

    def __len__(self) -> int:  # pragma: no cover - trivial
        """
        Get the number of bands.

        Returns:
            Number of bands in this signature set.
        """
        return len(self.bands)
