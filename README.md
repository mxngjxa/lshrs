# LSHRS

Redis-backed Locality Sensitive Hashing (LSH) helper that stores only bucket membership while delegating vector storage to external systems such as PostgreSQL.

## Installation

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
from lshrs import LSHRS

def fetch_vectors(indices):
    # Replace with your actual data store lookups
    vectors = np.load("vectors.npy")
    return vectors[indices]

lsh = LSHRS(
    dim=768,
    redis_host="localhost",
    num_perm=256,  # can be any positive value; bands/rows auto-tune via utils.br
    vector_fetch_fn=fetch_vectors,
)

# Stream vectors from PostgreSQL using the builtin loader.
# See lshrs/io/postgres.py for all available keyword arguments.
lsh.create_signatures(
    format="postgres",
    dsn="postgresql://user:pass@localhost/db",
    table="documents",
    index_column="doc_id",
    vector_column="embedding",
)

# Ad-hoc single inserts are also supported
lsh.ingest(42, np.random.randn(768).astype(np.float32))

query_vec = np.random.randn(768).astype(np.float32)

# Retrieve pure top-k collision candidates
top_candidates = lsh.get_top_k(query_vec, topk=10)

# Retrieve the top 20% of candidates with cosine reranking
reranked = lsh.get_above_p(query_vec, p=0.2)
```

## API Surface

- `LSHRS.create_signatures(format="postgres", **kwargs)`: Stream indexing from PostgreSQL or Parquet sources.
- `LSHRS.index(indices, vectors=None)`: Batch ingest vectors.
- `LSHRS.ingest(index, vector)`: Insert a single vector.
- `LSHRS.query(vector, top_k=10, top_p=None)`: Retrieve similar items.
- `LSHRS.get_top_k(vector, topk=10)`: Convenience alias for pure top-k retrieval.
- `LSHRS.get_above_p(vector, p=0.95)`: Convenience alias for reranked top-p retrieval.
- `LSHRS.delete(indices)`: Remove items from all buckets.
- `LSHRS.clear()`: Remove all keys for the configured prefix.
- `LSHRS.stats()`: Observe current configuration metadata.
- `LSHRS.save_to_disk(path) / LSHRS.load_from_disk(path, ...)`: Persist and restore configuration plus projection matrices.

## Design Notes

- Only Redis is used for bucket membership, vectors remain in your datastore.
- Automatic band/row parameter selection mirrors common LSH heuristics.
- Optional reranking with cosine similarity when `top_p` is requested.
- Minimal configuration with reasonable defaults to aid adoption.