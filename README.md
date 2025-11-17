# LSHRS

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/lshrs.svg)](https://pypi.org/project/lshrs/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

Redis-backed locality-sensitive hashing toolkit that stores bucket membership in Redis while keeping the heavy vector payloads in your primary datastore.

[![PyPI](https://github.com/mxngjxa/lshrs/actions/workflows/cd.yml/badge.svg)](https://github.com/mxngjxa/lshrs/actions/workflows/cd.yml)
[![Testing and Linting](https://github.com/mxngjxa/lshrs/actions/workflows/ci.yml/badge.svg)](https://github.com/mxngjxa/lshrs/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/lshrs.svg)](https://pypi.org/project/lshrs/)


<div align="center">
    <img src="docs/lshrs-logo.svg" alt="logo"></img>
</div>

## Table of Contents

- [Overview](##overview)
- [Architecture Snapshot](##architecture-snapshot)
- [Key Features](##key-features)
- [Installation](##installation)
- [Quick Start](##quick-start)
- [Ingestion Pipelines](##ingestion-pipelines)
- [Querying Modes](##querying-modes)
- [Persistence & Lifecycle](##persistence--lifecycle)
- [Performance & Scaling Guidelines](##performance--scaling-guidelines)
- [Troubleshooting](##troubleshooting)
- [API Surface Summary](##api-surface-summary)
- [Development & Testing](##development--testing)
- [License](##license)

## Overview

[`LSHRS`](lshrs/core/main.py:53) orchestrates the full locality-sensitive hashing (LSH) workflow:

1. Hash incoming vectors into stable banded signatures via random projections.
2. Store only bucket membership in Redis for low-latency candidate enumeration.
3. Optionally rerank candidates using cosine similarity with vectors fetched from your system of record.

The out-of-the-box configuration chooses bands/rows automatically, pipelines Redis operations, and exposes hooks for streaming data ingestion, persistence, and operational maintenance.

## Architecture Snapshot

| Concern | Component | Description |
| --- | --- | --- |
| Hashing | [`LSHHasher`](lshrs/hash/lsh.py:20) | Generates banded random-projection signatures. |
| Storage | [`RedisStorage`](lshrs/storage/redis.py:40) | Persists bucket membership using Redis sets and pipelines for batch writes. |
| Ingestion | [`LSHRS.create_signatures()`](lshrs/core/main.py:267) | Streams vectors from PostgreSQL or Parquet via pluggable loaders. |
| Reranking | [`top_k_cosine()`](lshrs/utils/similarity.py:94) | Computes cosine similarity for candidate reranking. |
| Configuration | [`get_optimal_config()`](lshrs/utils/br.py:326) | Picks band/row counts that match a target similarity threshold. |

## Key Features

- **Redis-native buckets**: Uses Redis sets for O(1) membership updates and pipelined batch ingestion.
- **Progressive indexing**: Stream vectors from PostgreSQL ([`iter_postgres_vectors()`](lshrs/io/postgres.py:16)) or Parquet ([`iter_parquet_vectors()`](lshrs/io/parquet.py:46)) without exhausting memory.
- **Dual retrieval modes**: Choose fast top-k collision lookups or cosine-reranked top-p filtering through [`LSHRS.query()`](lshrs/core/main.py:486).
- **Persistable hashing state**: Save and reload projection matrices with [`LSHRS.save_to_disk()`](lshrs/core/main.py:830) and [`LSHRS.load_from_disk()`](lshrs/core/main.py:881).
- **Operational safety**: Snapshot configuration with [`LSHRS.stats()`](lshrs/core/main.py:782), clear indices via [`LSHRS.clear()`](lshrs/core/main.py:755), and surgically delete members using [`LSHRS.delete()`](lshrs/core/main.py:710).

## Installation

### PyPI

```bash
uv install lshrs
```

Or, if installing for a postgres database:

```bash
uv install 'lshrs[postgres]'
```

### From source checkout

```bash
git clone https://github.com/mxngjxa/lshrs.git
cd lshrs
uv sync -e ".[dev]"
```

> [!NOTE]
> The project targets Python ≥ 3.13 as defined in [`pyproject.toml`](pyproject.toml).

### Optional extras

- PostgreSQL streaming requires [`psycopg`](https://www.psycopg.org/). Install with `uv add 'lshrs[postgres]'` or `uv add 'psycopg[binary]'`.
- Parquet ingestion requires [`pyarrow`](https://arrow.apache.org/). Install with `uv add pyarrow` or include it in your extras.

## Quick Start

```python
import numpy as np
from lshrs import lshrs

def fetch_vectors(indices: list[int]) -> np.ndarray:
    # Replace with your vector store retrieval (PostgreSQL, disk, object store, etc.)
    embeddings = np.load("vectors.npy")
    return embeddings[indices]

lsh = LSHRS(
    dim=768,
    num_perm=256,
    redis_host="localhost",
    redis_prefix="demo",
    vector_fetch_fn=fetch_vectors,
)

# Stream index construction from PostgreSQL
lsh.create_signatures(
    format="postgres",
    dsn="postgresql://user:pass@localhost/db",
    table="documents",
    index_column="doc_id",
    vector_column="embedding",
)

# Insert an ad-hoc document
lsh.ingest(42, np.random.randn(768).astype(np.float32))

# Retrieve candidates
query = np.random.randn(768).astype(np.float32)
top10 = lsh.get_top_k(query, topk=10)
reranked = lsh.get_above_p(query, p=0.2)
```

The code above exercises [`LSHRS.create_signatures()`](lshrs/core/main.py:267), [`LSHRS.ingest()`](lshrs/core/main.py:340), [`LSHRS.get_top_k()`](lshrs/core/main.py:626), and [`LSHRS.get_above_p()`](lshrs/core/main.py:661).

## Ingestion Pipelines

### Streaming from PostgreSQL

[`iter_postgres_vectors()`](lshrs/io/postgres.py:16) yields `(indices, vectors)` batches using server-side cursors:

```python
lsh.create_signatures(
    format="postgres",
    dsn="postgresql://reader:secret@analytics.db/search",
    table="embeddings",
    index_column="item_id",
    vector_column="embedding",
    batch_size=5_000,
    where_clause="updated_at >= NOW() - INTERVAL '1 day'",
)
```

> [!TIP]
> Provide a custom `connection_factory` if you need pooled connections or TLS configuration.

### Streaming from Parquet

[`iter_parquet_vectors()`](lshrs/io/parquet.py:46) supports memory-friendly batch loads from Parquet files:

```python
for ids, batch in iter_parquet_vectors(
    "captures/2024-01-embeddings.parquet",
    index_column="document_id",
    vector_column="embedding",
    batch_size=8_192,
):
    lsh.index(ids, batch)
```

> [!IMPORTANT]
> Install `pyarrow` prior to using the Parquet loader; otherwise [`iter_parquet_vectors()`](lshrs/io/parquet.py:46) raises `ImportError`.

### Manual or Buffered Ingestion

- [`LSHRS.index()`](lshrs/core/main.py:399) ingests vector batches you already hold in memory.
- [`LSHRS.ingest()`](lshrs/core/main.py:340) is ideal for realtime single-document updates.
- Under the hood, [`RedisStorage.batch_add()`](lshrs/storage/redis.py:340) leverages Redis pipelines for throughput.

## Querying Modes

[`LSHRS.query()`](lshrs/core/main.py:486) provides two complementary retrieval patterns:

| Mode | When to use | Result |
| --- | --- | --- |
| **Top-k** (`top_p=None`) | Latency-critical scenarios that only require coarse candidates. | Returns `List[int]` ordered by band collisions. |
| **Top-p** (`top_p=0.0–1.0`) | Precision-sensitive flows that can rerank using original vectors. | Returns `List[Tuple[int,float]]` of `(index, cosine_similarity)` pairs. |

> [!CAUTION]
> Reranking requires configuring `vector_fetch_fn` when instantiating [`LSHRS`](lshrs/core/main.py:53); otherwise top-p queries raise `RuntimeError`.

Supporting helpers:

- [`LSHRS.get_top_k()`](lshrs/core/main.py:626) wraps `query` for pure top-k retrieval.
- [`LSHRS.get_above_p()`](lshrs/core/main.py:661) wraps `query` with a similarity-mass cutoff.
- Cosine scoring is provided by [`cosine_similarity()`](lshrs/utils/similarity.py:25) and [`top_k_cosine()`](lshrs/utils/similarity.py:94).

## Persistence & Lifecycle

| Operation | Purpose | Reference |
| --- | --- | --- |
| Snapshot configuration | Inspect runtime parameters and Redis namespace. | [`LSHRS.stats()`](lshrs/core/main.py:782) |
| Flush & clear | Remove all Redis buckets for the configured prefix. | [`LSHRS.clear()`](lshrs/core/main.py:755) |
| Hard delete members | Remove specific indices across all buckets. | [`LSHRS.delete()`](lshrs/core/main.py:710) |
| Persist projections | Save configuration and projection matrices to disk. | [`LSHRS.save_to_disk()`](lshrs/core/main.py:830) |
| Restore projections | Rebuild an instance using saved matrices. | [`LSHRS.load_from_disk()`](lshrs/core/main.py:881) |

> [!WARNING]
> [`LSHRS.clear()`](lshrs/core/main.py:755) is irreversible—every key with the configured prefix is deleted. Back up state with [`LSHRS.save_to_disk()`](lshrs/core/main.py:830) beforehand if you need to rebuild.

## Performance & Scaling Guidelines

- **Choose sensible hash parameters**: [`get_optimal_config()`](lshrs/utils/br.py:326) finds bands/rows that approximate your target similarity threshold. Inspect S-curve behavior with [`compute_collision_probability()`](lshrs/utils/br.py:119).
- **Normalize inputs**: Pre-normalize vectors or rely on [`l2_norm()`](lshrs/utils/norm.py:4) for consistent cosine scores.
- **Batch ingestion**: When indexing large volumes, route operations through [`LSHRS.index()`](lshrs/core/main.py:399) to let [`RedisStorage.batch_add()`](lshrs/storage/redis.py:340) coalesce writes.
- **Monitor bucket sizes**: Large buckets indicate low selectivity. Adjust `num_perm`, `num_bands`, or the similarity threshold to trade precision vs. recall.
- **Pipeline warmup**: Flush outstanding operations with [`LSHRS._flush_buffer()`](lshrs/core/main.py:1177) (indirectly called) before measuring latency or persisting state.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `ImportError: psycopg is required` | PostgreSQL loader invoked without optional dependency. | Install `psycopg[binary]` or avoid `format="postgres"`. |
| `ValueError: Vectors must have shape (n, dim)` | Supplied batch dimension mismatched the configured `dim`. | Ensure all vectors match the `dim` passed to [`LSHRS.__init__()`](lshrs/core/main.py:149). |
| `ValueError: Cannot normalize zero vector` | Zero-length vectors were passed to cosine scoring utilities. | Filter zero vectors before reranking or normalize upstream. |
| Empty search results | Buckets never flushed to Redis. | Call [`LSHRS.index()`](lshrs/core/main.py:399) (auto flushes) or explicitly invoke [`LSHRS._flush_buffer()`](lshrs/core/main.py:1177) before querying. |
| Extremely large buckets | Similarity threshold too low / insufficient hash bits. | Increase `num_perm` or tweak target threshold via [`get_optimal_config()`](lshrs/utils/br.py:326). |

> [!TIP]
> Use Redis `SCAN` commands (e.g., `SCAN 0 MATCH lsh:*`) to inspect bucket distribution during tuning.

## API Surface Summary

| Area | Description | Primary Entry Point |
| --- | --- | --- |
| Ingestion orchestration | Bulk streaming with source-aware loaders. | [`LSHRS.create_signatures()`](lshrs/core/main.py:267) |
| Batch ingestion | Hash and store vectors already in memory. | [`LSHRS.index()`](lshrs/core/main.py:399) |
| Single ingestion | Add or update one vector id on the fly. | [`LSHRS.ingest()`](lshrs/core/main.py:340) |
| Candidate enumeration | General-purpose search with optional reranking. | [`LSHRS.query()`](lshrs/core/main.py:486) |
| Hash persistence | Save and restore LSH projection matrices. | [`LSHRS.save_to_disk()`](lshrs/core/main.py:830) / [`LSHRS.load_from_disk()`](lshrs/core/main.py:881) |
| Redis maintenance | Prefix-aware key deletion and batch removal. | [`RedisStorage.clear()`](lshrs/storage/redis.py:582) / [`RedisStorage.remove_indices()`](lshrs/storage/redis.py:411) |
| Probability utilities | Analyze band/row trade-offs and false rates. | [`compute_collision_probability()`](lshrs/utils/br.py:119) / [`compute_false_rates()`](lshrs/utils/br.py:161) |

## Development & Testing

1. Install development dependencies:

   ```bash
   uv add -e ".[dev]"
   ```

2. Run the test suite:

   ```bash
   uv run --dev pytest
   ```

3. Lint (if you have [`ruff`](https://github.com/astral-sh/ruff) configured):

   ```bash
   uv run --dev ruff check
   ```

> [!NOTE]
> Example snippets in this README are intended to be run under Python 3.13 with NumPy 2.x and Redis ≥ 7 as specified in [`pyproject.toml`](pyproject.toml).

## License

Licensed under the terms of [`LICENSE`](LICENSE).
