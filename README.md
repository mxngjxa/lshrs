# LSHRS

[![CI](https://github.com/mxngjxa/lshrs/actions/workflows/ci.yml/badge.svg)](https://github.com/mxngjxa/lshrs/actions/workflows/ci.yml) 
[![Publish to PyPI](https://github.com/mxngjxa/lshrs/actions/workflows/cd.yml/badge.svg)](https://github.com/mxngjxa/lshrs/actions/workflows/cd.yml) 
[![PyPI version](https://img.shields.io/pypi/v/lshrs.svg)](https://pypi.org/project/lshrs/) 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) 
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Redis-backed locality-sensitive hashing toolkit that stores bucket membership in Redis while keeping the heavy vector payloads in your primary datastore.


<div align="center">
    <img src="docs/lshrs-logo.svg" alt="logo"></img>
</div>

## Table of Contents

- [Positioning](#positioning)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Ingestion](#ingestion)
- [Querying](#querying)
- [Persistence & Lifecycle](#persistence--lifecycle)
- [API Surface](#api-surface)
- [Benchmarks](#benchmarks)
- [Development & Testing](#development--testing)
- [License](#license)

## Positioning

LSHRS adds cosine approximate-nearest-neighbor (ANN) search on top of the datastore you already run. Redis holds *only* a compact `bucket -> set-of-indices` index — typically much smaller than the vectors it indexes, with the gap widening as embedding dimensionality grows. The vectors themselves stay external in your system of record (PostgreSQL/pgvector, Parquet, an object store, anything) and are fetched on demand, only when you opt into cosine reranking via `vector_fetch_fn`.

See [docs/positioning.md](docs/positioning.md) for the full comparison (LSHRS vs datasketch vs RediSearch HNSW vs pgvector vs FAISS) and when *not* to use LSHRS.

## How It Works

[`LSHRS`](lshrs/core/main.py:53) runs the full LSH workflow: hash vectors into banded random-projection signatures, store only bucket membership in Redis for low-latency candidate enumeration, then optionally rerank candidates by cosine similarity using vectors fetched from your system of record. It auto-selects bands/rows, pipelines Redis operations, and exposes hooks for streaming ingestion, persistence, and maintenance.

| Concern | Component |
| --- | --- |
| Hashing | [`LSHHasher`](lshrs/hash/lsh.py:20) — banded random-projection signatures |
| Storage | [`RedisStorage`](lshrs/storage/redis.py:40) — bucket membership via Redis sets + pipelines |
| Ingestion | [`LSHRS.create_signatures()`](lshrs/core/main.py:267) — streams from PostgreSQL or Parquet |
| Reranking | [`top_k_cosine()`](lshrs/utils/similarity.py:94) — cosine similarity for candidates |
| Configuration | [`get_optimal_config()`](lshrs/utils/br.py:326) — bands/rows for a target threshold |

## Installation

```bash
pip install lshrs              # core
pip install 'lshrs[postgres]'  # + PostgreSQL streaming (psycopg)
pip install 'lshrs[parquet]'   # + Parquet ingestion (pyarrow)
```

From source:

```bash
git clone https://github.com/mxngjxa/lshrs.git
cd lshrs
uv sync --dev
```

> [!NOTE]
> Requires Python >= 3.10 (see [`pyproject.toml`](pyproject.toml)).

## Quick Start

```python
import numpy as np
from lshrs import LSHRS

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
top10 = lsh.get_top_k(query, topk=10)         # fast collision lookup -> List[int]
reranked = lsh.get_above_p(query, p=0.2)      # cosine-reranked -> List[(index, score)]
```

## Ingestion

- [`LSHRS.create_signatures()`](lshrs/core/main.py:267) streams batches from PostgreSQL ([`iter_postgres_vectors()`](lshrs/io/postgres.py:16), server-side cursors) or Parquet ([`iter_parquet_vectors()`](lshrs/io/parquet.py:46)). Tune `batch_size`, filter with `where_clause`, or pass a custom `connection_factory` for pooling/TLS.
- [`LSHRS.index()`](lshrs/core/main.py:399) ingests in-memory batches; [`LSHRS.ingest()`](lshrs/core/main.py:340) handles single realtime updates. Both coalesce writes through [`RedisStorage.batch_add()`](lshrs/storage/redis.py:340).

> [!IMPORTANT]
> Install `pyarrow` before using the Parquet loader, or [`iter_parquet_vectors()`](lshrs/io/parquet.py:46) raises `ImportError`.

## Querying

[`LSHRS.query()`](lshrs/core/main.py:486) offers two retrieval modes:

| Mode | When | Result |
| --- | --- | --- |
| **Top-k** (`top_p=None`) | Latency-critical, coarse candidates | `List[int]` ordered by band collisions |
| **Top-p** (`top_p=0.0–1.0`) | Precision-sensitive, rerank by cosine | `List[Tuple[int, float]]` of `(index, similarity)` |

Convenience wrappers: [`get_top_k()`](lshrs/core/main.py:626) and [`get_above_p()`](lshrs/core/main.py:661).

> [!CAUTION]
> Top-p reranking requires `vector_fetch_fn` at construction; otherwise it raises `RuntimeError`.

## Persistence & Lifecycle

| Operation | Reference |
| --- | --- |
| Inspect runtime config / Redis namespace | [`LSHRS.stats()`](lshrs/core/main.py:782) |
| Clear all buckets for the prefix (irreversible) | [`LSHRS.clear()`](lshrs/core/main.py:755) |
| Hard-delete specific indices | [`LSHRS.delete()`](lshrs/core/main.py:710) |
| Save / restore projection matrices | [`save_to_disk()`](lshrs/core/main.py:830) / [`load_from_disk()`](lshrs/core/main.py:881) |

Because LSHRS stores only `bucket -> set-of-indices` membership, [`LSHRS.delete()`](lshrs/core/main.py:710) makes per-id removal cheap and exact — an id is dropped from its bands with no rebuild or compaction, which suits GDPR / right-to-be-forgotten workflows. See [docs/positioning.md](docs/positioning.md#deletes--the-right-to-be-forgotten) for details.

> [!WARNING]
> [`LSHRS.clear()`](lshrs/core/main.py:755) deletes every key with the configured prefix. Back up with [`save_to_disk()`](lshrs/core/main.py:830) first if you need to rebuild.

## API Surface

| Area | Primary Entry Point |
| --- | --- |
| Bulk streaming ingestion | [`LSHRS.create_signatures()`](lshrs/core/main.py:267) |
| Batch / single ingestion | [`LSHRS.index()`](lshrs/core/main.py:399) / [`LSHRS.ingest()`](lshrs/core/main.py:340) |
| Search with optional reranking | [`LSHRS.query()`](lshrs/core/main.py:486) |
| Hash persistence | [`save_to_disk()`](lshrs/core/main.py:830) / [`load_from_disk()`](lshrs/core/main.py:881) |
| Redis maintenance | [`RedisStorage.clear()`](lshrs/storage/redis.py:582) / [`RedisStorage.remove_indices()`](lshrs/storage/redis.py:411) |
| Probability utilities | [`compute_collision_probability()`](lshrs/utils/br.py:119) / [`compute_false_rates()`](lshrs/utils/br.py:161) |

## Benchmarks

A reproducible benchmark harness lives in [`benchmarks/`](benchmarks/). It
generates synthetic random unit vectors and measures index build throughput,
top-k / top-p query latency (p50/p95), the Redis footprint ratio (raw
`N x dim x 4` bytes vs the bucket index — the ratio grows with `dim`), and
delete latency.

```bash
# Runs anywhere via in-memory fakeredis (functional timings, estimated footprint)
uv run python benchmarks/run_benchmark.py --n 5000 --dim 256 --num-perm 128

# Use a real Redis server for measured memory (MEMORY USAGE / INFO memory)
uv run python benchmarks/run_benchmark.py --real-redis --redis-host localhost
```

Results print as a markdown table and are written to
[`benchmarks/RESULTS_TEMPLATE.md`](benchmarks/RESULTS_TEMPLATE.md). See
[`benchmarks/README.md`](benchmarks/README.md) for full details. All numbers are
machine-dependent.

> [!TIP]
> [`get_optimal_config()`](lshrs/utils/br.py:326) picks bands/rows for a target similarity threshold. Large buckets signal low selectivity — raise `num_perm` or the threshold to trade recall for precision. Inspect distribution with `SCAN 0 MATCH lsh:*`.

## Development & Testing

```bash
git clone https://github.com/mxngjxa/lshrs.git
cd lshrs
uv sync --dev
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format --check .   # format check
```

## License

Licensed under the terms of [`LICENSE`](LICENSE).
