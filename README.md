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
    redis_host="localhost",
    num_perm=128,
    vector_fetch_fn=fetch_vectors,
)

all_indices = list(range(1_000_000))
lsh.index(all_indices)

query_vec = np.random.randn(768)
top_candidates = lsh.query(query_vec, top_k=10)
```

## API Surface

- `LSHRS.index(indices, vectors=None)`: Batch ingest vectors.
- `LSHRS.ingest(index, vector)`: Insert a single vector.
- `LSHRS.query(vector, top_k=10, top_p=None)`: Retrieve similar items.
- `LSHRS.delete(indices)`: Remove items from all buckets.
- `LSHRS.clear()`: Remove all keys for the configured prefix.
- `LSHRS.stats()`: Observe current configuration metadata.

## Design Notes

- Only Redis is used for bucket membership, vectors remain in your datastore.
- Automatic band/row parameter selection mirrors common LSH heuristics.
- Optional reranking with cosine similarity when `top_p` is requested.
- Minimal configuration with reasonable defaults to aid adoption.