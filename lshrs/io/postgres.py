from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

DEFAULT_POSTGRES_BATCH_SIZE = 10_000

try:  # pragma: no cover - optional dependency
    import psycopg  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]


def iter_postgres_vectors(
    *,
    dsn: Optional[str] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    table: str = "vectors",
    index_column: str = "id",
    vector_column: str = "embedding",
    batch_size: int = DEFAULT_POSTGRES_BATCH_SIZE,
    limit: Optional[int] = None,
    where_clause: Optional[str] = None,
    order_by: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
    fetch_query: Optional[str] = None,
) -> Iterator[Tuple[List[int], NDArray[np.float32]]]:
    """
    Stream ``(indices, vectors)`` pairs from PostgreSQL.

    Parameters
    ----------
    dsn:
        Data source name passed directly to :func:`psycopg.connect`. Either ``dsn`` or
        ``connection_factory`` must be supplied.
    connection_factory:
        Callable returning an open psycopg connection. The caller retains ownership of
        the connection and is responsible for closing it.
    table:
        Name of the table containing the vectors when ``fetch_query`` is not provided.
    index_column:
        Name of the integer identifier column.
    vector_column:
        Name of the column storing vector embeddings.
    batch_size:
        Number of rows to fetch per iteration from the server-side cursor.
    limit:
        Optional maximum number of rows to retrieve.
    where_clause:
        Optional SQL snippet appended after ``WHERE``. The caller is responsible for
        parameterisation and sanitisation when using raw SQL fragments.
    order_by:
        Optional SQL snippet appended after ``ORDER BY`` (e.g. ``"id ASC"``).
    params:
        Positional parameters passed to ``fetch_query`` when supplied.
    fetch_query:
        Fully customised SQL query. When provided, ``table``/``index_column``/
        ``vector_column``/``where_clause``/``order_by``/``limit`` are ignored.

    Yields
    ------
    Tuple[List[int], NDArray[np.float32]]
        A pair containing the list of indices and a ``(n, dim)`` float32 matrix.

    Raises
    ------
    ImportError
        If :mod:`psycopg` is not installed.
    ValueError
        If neither ``dsn`` nor ``connection_factory`` is provided, or if ``batch_size``
        is invalid, or when vector dimensionality is inconsistent.
    """
    if psycopg is None:
        raise ImportError(
            "psycopg is required to stream data from PostgreSQL. "
            "Install it via `pip install psycopg[binary]`."
        )

    if connection_factory is None and dsn is None:
        raise ValueError("Either `dsn` or `connection_factory` must be provided")

    if fetch_query is None and params is not None:
        raise ValueError("`params` can only be used when `fetch_query` is supplied")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    owned_connection = False
    if connection_factory is not None:
        connection = connection_factory()
    else:
        assert dsn is not None  # MyPy guard
        connection = psycopg.connect(dsn)  # type: ignore[assignment]
        connection.autocommit = True
        owned_connection = True

    try:
        query, query_params = _build_query(
            fetch_query=fetch_query,
            table=table,
            index_column=index_column,
            vector_column=vector_column,
            limit=limit,
            where_clause=where_clause,
            order_by=order_by,
            batch_params=params,
        )

        with connection.cursor(name="lshrs_stream") as cursor:
            cursor.itersize = batch_size
            cursor.execute(query, query_params)

            expected_dim: Optional[int] = None

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                indices: List[int] = []
                vectors: List[NDArray[np.float32]] = []

                for row_idx, row in enumerate(rows):
                    idx = int(row[0])
                    vector = _coerce_vector(row[1])

                    if expected_dim is None:
                        expected_dim = vector.shape[0]
                    elif vector.shape[0] != expected_dim:
                        raise ValueError(
                            "Inconsistent vector dimensionality detected while "
                            "streaming from PostgreSQL: "
                            f"expected {expected_dim}, received {vector.shape[0]}"
                        )

                    indices.append(idx)
                    vectors.append(vector)

                yield indices, np.stack(vectors, axis=0).astype(np.float32, copy=False)
    finally:
        if owned_connection:
            connection.close()


def _build_query(
    *,
    fetch_query: Optional[str],
    table: str,
    index_column: str,
    vector_column: str,
    limit: Optional[int],
    where_clause: Optional[str],
    order_by: Optional[str],
    batch_params: Optional[Sequence[Any]],
) -> Tuple[Any, Tuple[Any, ...]]:
    """
    Construct the SQL query and parameter tuple for streaming vector data.
    """
    if fetch_query is not None:
        return fetch_query, tuple(batch_params or ())

    from psycopg import sql  # Imported lazily to avoid import when psycopg missing

    query = sql.SQL("SELECT {index}, {vector} FROM {table}").format(
        index=sql.Identifier(index_column),
        vector=sql.Identifier(vector_column),
        table=sql.Identifier(table),
    )

    if where_clause:
        query += sql.SQL(" WHERE ") + sql.SQL(where_clause)

    if order_by:
        query += sql.SQL(" ORDER BY ") + sql.SQL(order_by)

    params: List[Any] = []
    if limit is not None:
        query += sql.SQL(" LIMIT %s")
        params.append(limit)

    return query, tuple(params)


def _coerce_vector(raw_value: Any) -> NDArray[np.float32]:
    """
    Convert PostgreSQL cell data into a flat float32 numpy array.
    """
    if isinstance(raw_value, memoryview):
        array = np.frombuffer(raw_value.tobytes(), dtype=np.float32)
    elif isinstance(raw_value, (bytes, bytearray)):
        array = np.frombuffer(raw_value, dtype=np.float32)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip("{}[]() ")
        if not stripped:
            raise ValueError(
                "Encountered empty vector representation in PostgreSQL row"
            )
        array = np.fromiter(
            (float(part) for part in stripped.split(",")),
            dtype=np.float32,
        )
    else:
        array = np.asarray(raw_value, dtype=np.float32).reshape(-1)

    if array.size == 0:
        raise ValueError("Encountered empty vector while decoding PostgreSQL row")

    return array.reshape(-1).astype(np.float32, copy=False)


__all__ = ["iter_postgres_vectors", "DEFAULT_POSTGRES_BATCH_SIZE"]
