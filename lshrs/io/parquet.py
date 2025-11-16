"""
Parquet File Vector Loader for LSH Indexing

This module provides efficient streaming capabilities for loading vector embeddings
from Apache Parquet files. It's designed to handle large-scale vector datasets that
don't fit in memory by using incremental batch processing.

Key features:
    - Memory-efficient streaming using PyArrow's iterator API
    - Automatic type conversion to numpy float32 arrays
    - Configurable batch sizes for memory/speed tradeoffs
    - Validation of vector dimensions and data integrity

Parquet format advantages for vector storage:
    - Columnar storage is efficient for large embedding arrays
    - Built-in compression reduces disk space
    - Schema enforcement ensures data consistency
    - Supports nested arrays/lists for vector storage
    - Excellent integration with data processing frameworks

Typical usage:
    >>> for indices, vectors in iter_parquet_vectors("embeddings.parquet"):
    ...     lsh.index(indices, vectors)  # Process batch
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Conditional import of PyArrow (optional dependency)
# This allows the package to be installed without pyarrow for users who don't need Parquet support
try:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pq = None  # type: ignore[assignment]

# Default batch size balances memory usage with I/O efficiency
# 10,000 vectors × 768 dims × 4 bytes ≈ 30 MB per batch (reasonable for most systems)
DEFAULT_PARQUET_BATCH_SIZE = 10_000


def iter_parquet_vectors(
    source: Path | str,
    *,
    index_column: str = "index",
    vector_column: str = "vector",
    batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[Tuple[List[int], NDArray[np.float32]]]:
    """
    Stream (indices, vectors) pairs from a Parquet file in batches.

    This function provides memory-efficient loading of vector embeddings stored
    in Parquet format. It uses PyArrow's columnar reader to stream data in
    configurable batches, enabling processing of datasets larger than RAM.

    The function expects a Parquet file with:
        1. An integer column containing unique identifiers
        2. An array/list column containing vector embeddings

    Common Parquet schemas for vectors:
        - Fixed-size lists: LIST<FLOAT32> with fixed length
        - Variable lists: LIST<FLOAT32> (validated for consistency)
        - Nested arrays: Compatible with pandas/numpy array storage

    Memory usage: O(batch_size × vector_dimension × 4 bytes)

    Parameters
    ----------
    source : Path or str
        Path to the Parquet file on disk. Can be absolute or relative.
        Tilde expansion (~) is supported for home directories.

    index_column : str, default="index"
        Name of the column containing integer identifiers.
        These become the keys in your LSH index.
        Common names: "id", "index", "doc_id", "item_id"

    vector_column : str, default="vector"
        Name of the column containing vector embeddings.
        Vectors must be stored as arrays or lists of floats.
        Common names: "embedding", "vector", "features", "representation"

    batch_size : int, default=10000
        Number of rows to read per iteration.

        Trade-offs:
            - Larger: Better I/O efficiency, more memory usage
            - Smaller: Less memory, more I/O operations

        Guidelines:
            - Small vectors (128d): 10,000-50,000 rows
            - Medium vectors (768d): 5,000-20,000 rows
            - Large vectors (4096d): 1,000-5,000 rows

    Yields
    ------
    Iterator[Tuple[List[int], NDArray[np.float32]]]
        For each batch yields a tuple of:
            - indices: List of integer IDs from index_column
            - vectors: 2D numpy array of shape (batch_size, dimension)

        Last batch may have fewer rows than batch_size.
        All vectors are converted to float32 for consistency.

    Raises
    ------
    ImportError
        If pyarrow is not installed. Install with: pip install pyarrow

    FileNotFoundError
        If the Parquet file doesn't exist at the specified path.

    ValueError
        If required columns are missing from the Parquet schema.
        If batch_size is <= 0.
        If vectors have inconsistent dimensions within or across batches.
        If any vector is empty (zero-length).

    Examples
    --------
    Basic usage with default column names:

    >>> for indices, vectors in iter_parquet_vectors("data.parquet"):
    ...     print(f"Batch: {len(indices)} vectors of shape {vectors.shape}")
    ...     # Process batch...

    Custom columns and batch size:

    >>> loader = iter_parquet_vectors(
    ...     "embeddings.parquet",
    ...     index_column="document_id",
    ...     vector_column="bert_embedding",
    ...     batch_size=5000
    ... )
    >>> for batch_indices, batch_vectors in loader:
    ...     lsh.index(batch_indices, batch_vectors)

    Integration with LSH indexing:

    >>> lsh = LSHRS(dim=768)
    >>> total = 0
    >>> for indices, vectors in iter_parquet_vectors("corpus.parquet"):
    ...     lsh.index(indices, vectors)
    ...     total += len(indices)
    >>> print(f"Indexed {total} vectors")

    Notes
    -----
    File format requirements:
        - Must be valid Apache Parquet format
        - Index column must contain integers (int32/int64)
        - Vector column must contain arrays/lists of numbers
        - All vectors must have the same dimension

    Performance tips:
        - Use columnar selection (only reads needed columns)
        - Parquet files with row group sizes matching batch_size are optimal
        - Consider sorting by index column for better locality
        - Use Snappy or LZ4 compression for balance of speed/size

    Memory management:
        - Each batch allocates new memory (previous batches can be GC'd)
        - Peak memory ≈ batch_size × vector_dim × 4 bytes
        - For 768-dim vectors: 10k batch = ~30MB, 100k = ~300MB
    """
    # Check if PyArrow is available (optional dependency)
    if pq is None:
        raise ImportError(
            "pyarrow is required to stream vectors from Parquet files. "
            "Install it via `pip install pyarrow`."
        )

    # Resolve path with tilde expansion for home directory
    # expanduser() converts ~/data.parquet → /home/user/data.parquet
    path = Path(source).expanduser()

    # Validate file existence with clear error message
    if not path.exists():
        raise FileNotFoundError(f"Parquet source '{path}' does not exist")

    # Validate batch size parameter
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    # Open Parquet file for reading
    # ParquetFile provides low-level access with iterator support
    parquet_file = pq.ParquetFile(path)

    # Get Arrow schema for column validation
    # schema_arrow is the PyArrow schema (not Parquet schema)
    schema = parquet_file.schema_arrow

    # Validate that required columns exist in the schema
    # get_field_index returns -1 if column doesn't exist
    for column in (index_column, vector_column):
        if schema.get_field_index(column) == -1:
            raise ValueError(
                f"Column '{column}' was not found in Parquet schema {schema.names}"
            )

    # Stream file in batches for memory efficiency
    # iter_batches yields RecordBatch objects with columnar data
    # Only specified columns are read (columnar projection)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=[index_column, vector_column]
    ):
        # Skip empty batches (can occur at file boundaries)
        if batch.num_rows == 0:
            continue

        # Extract columns by position (more efficient than by name)
        # Column 0 = index_column, Column 1 = vector_column
        indices_array = batch.column(0)
        vectors_array = batch.column(1)

        # Convert Arrow arrays to Python lists
        # to_pylist() handles Arrow → Python type conversion
        # Cast indices to int (handles int32/int64/uint variants)
        indices = [int(value) for value in indices_array.to_pylist()]

        # Convert vector lists to numpy array with validation
        # _coerce_vectors handles dimension checking and type conversion
        vectors = _coerce_vectors(vectors_array.to_pylist())

        # Yield batch for processing
        # Caller receives clean Python types ready for use
        yield indices, vectors


def _coerce_vectors(rows: Sequence[Sequence[float]]) -> NDArray[np.float32]:
    """
    Convert a sequence of vector rows into a dense float32 matrix.

    This helper function performs critical validation and normalization:
        1. Converts each row to numpy array
        2. Validates non-empty vectors
        3. Ensures dimensional consistency
        4. Stacks into efficient 2D matrix

    The function enforces that all vectors in a batch have identical
    dimensions, which is required for efficient matrix operations in
    LSH and similarity computations.

    Parameters
    ----------
    rows : Sequence[Sequence[float]]
        List of vector rows from Parquet file.
        Each row is a list/array of floating point values.
        Can be Python lists, numpy arrays, or other sequences.

    Returns
    -------
    NDArray[np.float32]
        2D array of shape (num_vectors, dimension).
        All vectors normalized to float32 dtype.
        Contiguous memory layout for cache efficiency.

    Raises
    ------
    ValueError
        If any vector is empty (zero length).
        If vectors have inconsistent dimensions within the batch.

    Examples
    --------
    >>> rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    >>> matrix = _coerce_vectors(rows)
    >>> matrix.shape
    (2, 3)
    >>> matrix.dtype
    dtype('float32')

    Notes
    -----
    Design decisions:
        - float32 is used for memory efficiency (vs float64)
        - Dimension validation happens per-batch (not globally)
        - Empty vectors raise errors (not silently skipped)
        - Uses numpy.stack for efficient memory layout

    Performance characteristics:
        - O(n × d) time where n = num_vectors, d = dimension
        - Single allocation for output matrix
        - Vectorized operations where possible
    """
    # Accumulator for normalized vectors
    normalized: List[NDArray[np.float32]] = []

    # Track expected dimension (set from first vector)
    # None initially, then locked to first vector's dimension
    expected_dim: Optional[int] = None

    # Process each row into normalized numpy array
    for row in rows:
        # Convert to numpy array and ensure 1D shape
        # asarray doesn't copy if already numpy array
        # reshape(-1) flattens any nested structure
        arr = np.asarray(row, dtype=np.float32).reshape(-1)

        # Validate non-empty vector
        # Empty vectors break similarity computations
        if arr.size == 0:
            raise ValueError("Encountered empty vector while reading Parquet data")

        # Set expected dimension from first vector
        if expected_dim is None:
            expected_dim = arr.shape[0]
        # Validate subsequent vectors match dimension
        elif arr.shape[0] != expected_dim:
            raise ValueError(
                "All vectors must share the same dimensionality; "
                f"expected {expected_dim}, received {arr.shape[0]}"
            )

        # Add validated vector to collection
        normalized.append(arr)

    # Stack vectors into efficient 2D matrix
    # axis=0 stacks along first dimension (rows)
    # Result is contiguous memory for better cache performance
    return np.stack(normalized, axis=0)
