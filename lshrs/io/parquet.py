from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pq = None  # type: ignore[assignment]

DEFAULT_PARQUET_BATCH_SIZE = 10_000


def iter_parquet_vectors(
    source: Path | str,
    *,
    index_column: str = "index",
    vector_column: str = "vector",
    batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[Tuple[List[int], NDArray[np.float32]]]:
    """
    Stream ``(indices, vectors)`` pairs from a Parquet file.

    The file is read incrementally using ``pyarrow`` so that large datasets can be
    processed without loading the entire table into memory at once.

    Parameters
    ----------
    source:
        Path to the Parquet file on disk.
    index_column:
        Name of the column containing integer identifiers.
    vector_column:
        Name of the column containing vector embeddings stored as arrays/lists.
    batch_size:
        Number of rows to read per iteration. Larger values trade memory for throughput.

    Yields
    ------
    Iterator[Tuple[List[int], NDArray[np.float32]]]
        Tuples containing a list of integer indices and an ``(n, dim)`` float32 array.

    Raises
    ------
    ImportError
        If ``pyarrow`` is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file schema is incompatible or contains malformed vectors.
    """
    if pq is None:
        raise ImportError(
            "pyarrow is required to stream vectors from Parquet files. "
            "Install it via `pip install pyarrow`."
        )

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Parquet source '{path}' does not exist")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow

    for column in (index_column, vector_column):
        if schema.get_field_index(column) == -1:
            raise ValueError(
                f"Column '{column}' was not found in Parquet schema {schema.names}"
            )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=[index_column, vector_column]
    ):
        if batch.num_rows == 0:
            continue

        indices_array = batch.column(0)
        vectors_array = batch.column(1)

        indices = [int(value) for value in indices_array.to_pylist()]
        vectors = _coerce_vectors(vectors_array.to_pylist())

        yield indices, vectors


def _coerce_vectors(rows: Sequence[Sequence[float]]) -> NDArray[np.float32]:
    """
    Convert a sequence of Python iterables into a dense float32 matrix.
    """
    normalized: List[NDArray[np.float32]] = []
    expected_dim: Optional[int] = None

    for row in rows:
        arr = np.asarray(row, dtype=np.float32).reshape(-1)

        if arr.size == 0:
            raise ValueError("Encountered empty vector while reading Parquet data")

        if expected_dim is None:
            expected_dim = arr.shape[0]
        elif arr.shape[0] != expected_dim:
            raise ValueError(
                "All vectors must share the same dimensionality; "
                f"expected {expected_dim}, received {arr.shape[0]}"
            )

        normalized.append(arr)

    return np.stack(normalized, axis=0)