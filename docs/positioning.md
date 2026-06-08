# Why LSHRS / Positioning

**LSHRS adds cosine approximate-nearest-neighbor (ANN) search to the datastore you already run — PostgreSQL/pgvector, Parquet, an object store, anything — without copying your vectors into a dedicated vector database.** Redis holds *only* a compact `bucket -> set-of-indices` index. The heavy vector payloads never leave your system of record; they are fetched on demand, and only when you opt into cosine reranking via `vector_fetch_fn`.

This is a deliberate inversion of the usual Redis-LSH design. Tools like NearPy also do cosine random-projection LSH over Redis, but they store the *full vectors* in Redis (and the project is unmaintained). Redis Stack's RediSearch likewise stores vectors in Redis (FLAT/HNSW). LSHRS goes the other way: the index is tiny and lives in Redis, the data stays where it already is.

The market already accepts the "Redis-backed LSH index" pattern — [datasketch](https://github.com/ekzhu/datasketch), the dominant Python LSH library, is built on it. But datasketch is Jaccard/MinHash-only and has no cosine/SimHash support. **LSHRS fills the cosine slot in that same familiar, Redis-backed style.**

**Why the footprint matters.** Redis stores only bucket membership, which is much smaller than the raw vectors — and the gap widens with embedding dimensionality (index cost is driven by `num_bands`, while each vector costs `dim x 4` bytes). The advantage is real but configuration-dependent: at low dimensions with many bands it is modest (our harness measures ~1.6x at dim=256 with the default 8 bands); at high dimensions (1k+ dim embeddings) it is large. As an illustration of the asymmetry, a [ClickHouse writeup](https://clickhouse.com/blog/approximate-nearest-neighbour-ann-with-sql-powered-local-sensitive-hashing-lsh-random-projections) measured ~13.67 MiB of LSH bits against ~1.48 GiB of raw vectors for 2.1M x 300-dim embeddings (~100x) — cited as evidence for the design principle, not as a benchmark of LSHRS itself. Measure your own configuration with the harness in [`benchmarks/`](../benchmarks/); see [requirements.md](../requirements.md) for benchmark status.

## Comparison

| | **LSHRS** | datasketch | RediSearch (HNSW) | pgvector | FAISS |
| --- | --- | --- | --- | --- | --- |
| Similarity metric | Cosine (RP-LSH) | Jaccard / MinHash only | Cosine, L2, IP | Cosine, L2, IP | Cosine, L2, IP |
| Where vectors live | **Your existing datastore** | Your storage (index in Redis) | In Redis | In PostgreSQL | In FAISS index / on-disk (FAISS format) |
| Redis footprint | **Index only (≪ raw vectors; ratio grows with `dim`)** | Index only | Full vectors | n/a | n/a |
| Data migration needed | **None — BYO datastore** | None | Yes (load vectors into Redis) | Vectors must live in PG | Yes (build/maintain FAISS index) |
| Delete cost | **Cheap & exact** (remove id from its bands) | Cheap (MinHash LSH) | Supported | Supported (SQL `DELETE`) | Varies by index type |
| Best-fit use case | Add cosine ANN to an existing datastore with a tiny Redis index and easy deletes | Jaccard dedup / near-duplicate detection | Max recall/QPS with Redis as the store | ANN inside an existing Postgres | Large-scale in-process ANN |

> [!NOTE]
> Adjacent designs solve different problems. FAISS `OnDiskInvertedLists` does separate the payload from the index, but in FAISS's *own* on-disk format — not your existing system of record. LSHRS keeps the vectors in the datastore you already operate.

## When NOT to use LSHRS

LSHRS optimizes for a small Redis footprint, no data migration, and cheap exact deletes — not for peak recall or throughput. If you need maximum recall and QPS at scale, reach for a dedicated vector database or RediSearch's HNSW index, which keep vectors resident in the engine and are tuned for that. LSHRS is the right fit when you would rather leave your vectors in your existing datastore and keep only a lightweight index in Redis.

## Deletes & the right to be forgotten

Because LSHRS stores only `bucket -> set-of-indices` membership, [`LSHRS.delete()`](../lshrs/core/main.py) makes per-id removal **cheap and exact**: an id is simply dropped from the bands it belongs to, with no index rebuild or compaction. This maps directly onto GDPR / "right to be forgotten" and machine-unlearning workflows (cf. USR-LSH, 2023), where individual records must be removed on demand:

```python
# Remove a single id (e.g. a GDPR erasure request) from the index.
lsh.delete(42)

# Remove a batch of ids in one call.
lsh.delete([42, 1337, 9001])
```

The id is gone from the LSH index immediately; you remain responsible for deleting the corresponding vector and source record from your own datastore.
