# LSHRS API Documentation

High-performance Redis-backed Locality Sensitive Hashing for approximate nearest neighbor search.

## Core Classes

### LSHRS

Main orchestrator class for Redis-backed Locality Sensitive Hashing pipeline.

```python
class LSHRS(
    dim: int,
    num_perm: int = 128,
    num_bands: int = None,
    rows_per_band: int = None,
    similarity_threshold: float = 0.5,
    buffer_size: int = 10000,
    vector_fetch_fn: callable = None,
    storage: RedisStorage = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str = None,
    redis_prefix: str = "lsh",
    decode_responses: bool = False,
    seed: int = 42
)
```

**Parameters:**
- `dim`: Dimensionality of vectors being indexed (fixed after initialization)
- `num_perm`: Total number of random projections/hash bits (default: 128)
- `num_bands`: Number of independent hash bands/tables (auto-computed if not specified)
- `rows_per_band`: Number of hash bits per band (auto-computed if not specified)
- `similarity_threshold`: Target Jaccard similarity for automatic band/row selection (default: 0.5)
- `buffer_size`: Number of Redis operations to accumulate before pipelining (default: 10000)
- `vector_fetch_fn`: Function to retrieve vectors from primary storage for reranking
- `storage`: Pre-configured Redis storage instance (optional)
- `redis_host`: Redis server hostname (default: "localhost")
- `redis_port`: Redis server port (default: 6379)
- `redis_db`: Redis database number (default: 0)
- `redis_password`: Redis authentication password (optional)
- `redis_prefix`: Key prefix for all Redis operations (default: "lsh")
- `decode_responses`: Whether Redis should decode responses to strings (default: False)
- `seed`: Random seed for projection matrix generation (default: 42)

## Core Methods

### ingest()

Add a single vector to the LSH index.

```python
def ingest(index: int, vector: np.ndarray) -> None
```

**Parameters:**
- `index`: Unique identifier for the vector
- `vector`: Input vector of dimension `dim`

**Example:**
```python
lsh.ingest(42, np.random.randn(768).astype(np.float32))
```

### index()

Batch index multiple vectors at once.

```python
def index(indices: List[int], vectors: np.ndarray = None) -> None
```

**Parameters:**
- `indices`: List of unique identifiers for vectors
- `vectors`: Array of shape `(n, dim)` or None if using `vector_fetch_fn`

**Example:**
```python
indices = [1, 2, 3, 4, 5]
vectors = np.random.randn(5, 768).astype(np.float32)
lsh.index(indices, vectors)
```

### create_signatures()

Stream and index vectors from various data sources.

```python
def create_signatures(
    format: str,
    dsn: str = None,
    table: str = None,
    index_column: str = None,
    vector_column: str = None,
    batch_size: int = 10000,
    where_clause: str = None,
    connection_factory: callable = None,
    file_path: str = None
) -> None
```

**Parameters:**
- `format`: Data source format ("postgres" or "parquet")
- `dsn`: Database connection string (for PostgreSQL)
- `table`: Table name containing vectors
- `index_column`: Column name for vector IDs
- `vector_column`: Column name for vector data
- `batch_size`: Number of vectors per batch (default: 10000)
- `where_clause`: SQL WHERE clause for filtering (optional)
- `connection_factory`: Custom connection factory (optional)
- `file_path`: Path to Parquet file (for Parquet format)

**Example:**
```python
# PostgreSQL streaming
lsh.create_signatures(
    format="postgres",
    dsn="postgresql://user:pass@localhost/db",
    table="documents",
    index_column="doc_id",
    vector_column="embedding",
    where_clause="created_at >= '2024-01-01'"
)

# Parquet streaming
lsh.create_signatures(
    format="parquet",
    file_path="embeddings.parquet",
    index_column="id",
    vector_column="vector"
)
```

### query()

Search for similar vectors with optional reranking.

```python
def query(
    vector: np.ndarray,
    topk: int = 10,
    topp: float = None
) -> Union[List[int], List[Tuple[int, float]]]
```

**Parameters:**
- `vector`: Query vector of dimension `dim`
- `topk`: Maximum number of candidates to return (default: 10)
- `topp`: Cumulative proportion threshold for similarity-based reranking (optional)

**Returns:**
- Top-k mode: List of candidate indices sorted by band collisions
- Top-p mode: List of (index, cosine_similarity) tuples sorted by similarity

**Example:**
```python
# Fast top-k retrieval without reranking
candidates = lsh.query(query_vec, topk=20)

# Accurate top-p retrieval with cosine reranking
scores = lsh.query(query_vec, topp=0.05, topk=100)
```

### get_top_k()

Convenience wrapper for pure top-k retrieval without reranking.

```python
def get_top_k(vector: np.ndarray, topk: int = 10) -> List[int]
```

**Parameters:**
- `vector`: Query vector of dimension `dim`
- `topk`: Number of top candidates to return

**Returns:**
- List of candidate indices sorted by band collision count

**Example:**
```python
top10 = lsh.get_top_k(query_vec, topk=10)
```

### get_above_p()

Convenience wrapper for top-p retrieval with cosine similarity reranking.

```python
def get_above_p(vector: np.ndarray, p: float = 0.95) -> List[Tuple[int, float]]
```

**Parameters:**
- `vector`: Query vector of dimension `dim`
- `p`: Proportion threshold in [0, 1] for similarity mass

**Returns:**
- List of (index, cosine_similarity) pairs sorted by similarity

**Example:**
```python
# Get top 5% most similar vectors
results = lsh.get_above_p(query_vec, p=0.05)
indices = [idx for idx, score in results]
```

### delete()

Remove vector indices from all Redis buckets.

```python
def delete(indices: Union[int, List[int]]) -> None
```

**Parameters:**
- `indices`: Vector index/indices to remove from the LSH index

**Example:**
```python
# Delete single vector
lsh.delete(42)

# Delete multiple vectors
lsh.delete([10, 20, 30])
```

### clear()

Delete all keys under the configured prefix (destructive operation).

```python
def clear() -> None
```

**Warning:** This operation is irreversible. All bucket data will be permanently deleted.

**Example:**
```python
# Clear entire index
lsh.clear()
```

### stats()

Get current configuration and statistics.

```python
def stats() -> Dict[str, Any]
```

**Returns:**
- Dictionary containing configuration parameters and index statistics

**Example:**
```python
info = lsh.stats()
print(f"Dimensions: {info['dim']}")
print(f"Bands: {info['num_bands']}")
print(f"Rows per band: {info['rows_per_band']}")
```

### save_to_disk()

Save LSH configuration and projection matrices to disk.

```python
def save_to_disk(path: str) -> None
```

**Parameters:**
- `path`: File path to save the configuration

**Example:**
```python
lsh.save_to_disk("model.lsh")
```

### load_from_disk()

Load LSH configuration and projection matrices from disk.

```python
@classmethod
def load_from_disk(path: str, **kwargs) -> LSHRS
```

**Parameters:**
- `path`: File path to load configuration from
- `**kwargs`: Additional parameters to override saved configuration

**Returns:**
- New LSHRS instance with loaded configuration

**Example:**
```python
# Load with same Redis configuration
lsh = LSHRS.load_from_disk("model.lsh")

# Load with different Redis host
lsh = LSHRS.load_from_disk("model.lsh", redis_host="new-host")
```

### _flush_buffer()

Internal method to flush pending Redis operations (called automatically).

```python
def _flush_buffer() -> None
```

**Note:** This is typically called automatically during `index()` operations and before queries. Manual invocation is rarely needed.

## Supporting Classes

### LSHHasher

Random projection based Locality-Sensitive Hashing hasher.

```python
class LSHHasher(
    num_bands: int,
    rows_per_band: int,
    dim: int,
    seed: int = 42
)
```

**Methods:**
- `hash_vector(vector: np.ndarray) -> HashSignatures`: Hash a single vector into LSH band signatures

### RedisStorage

Redis backend for storing LSH bucket membership.

```python
class RedisStorage(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str = None,
    prefix: str = "lsh",
    decode_responses: bool = False
)
```

**Methods:**
- `add_to_bucket(band_id: int, signature: bytes, index: int)`: Add index to bucket
- `get_bucket(band_id: int, signature: bytes) -> Set[int]`: Get indices from bucket
- `batch_add(operations: List[BucketOperation])`: Batch add operations
- `remove_indices(indices: List[int])`: Remove indices from all buckets
- `clear()`: Delete all keys with configured prefix

## Utility Functions

### get_optimal_config()

Find optimal band/row configuration for target similarity threshold.

```python
def get_optimal_config(
    num_perm: int,
    similarity_threshold: float = 0.5
) -> Tuple[int, int]
```

**Parameters:**
- `num_perm`: Total number of hash functions
- `similarity_threshold`: Target Jaccard similarity threshold

**Returns:**
- Tuple of (num_bands, rows_per_band)

### compute_collision_probability()

Compute probability of hash collision for given similarity.

```python
def compute_collision_probability(
    similarity: float,
    num_bands: int,
    rows_per_band: int
) -> float
```

**Parameters:**
- `similarity`: Jaccard similarity between vectors
- `num_bands`: Number of LSH bands
- `rows_per_band`: Number of rows per band

**Returns:**
- Probability of at least one band collision

### top_k_cosine()

Find top-k most similar vectors using cosine similarity.

```python
def top_k_cosine(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int
) -> List[Tuple[int, float]]
```

**Parameters:**
- `query`: Query vector
- `candidates`: Candidate vectors array
- `k`: Number of top results to return

**Returns:**
- List of (index, similarity) tuples sorted by similarity

### cosine_similarity()

Compute cosine similarity between query and candidate vectors.

```python
def cosine_similarity(
    query: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray
```

**Parameters:**
- `query`: Query vector
- `candidates`: Array of candidate vectors

**Returns:**
- Array of cosine similarity scores

### l2_norm()

Normalize vectors to unit length.

```python
def l2_norm(vectors: np.ndarray) -> np.ndarray
```

**Parameters:**
- `vectors`: Input vectors

**Returns:**
- L2-normalized vectors

## Data Loaders

### iter_postgres_vectors()

Stream vectors from PostgreSQL using server-side cursors.

```python
def iter_postgres_vectors(
    dsn: str,
    table: str,
    index_column: str,
    vector_column: str,
    batch_size: int = 10000,
    where_clause: str = None,
    connection_factory: callable = None
) -> Iterator[Tuple[List[int], np.ndarray]]
```

**Yields:**
- Tuples of (indices, vectors) batches

### iter_parquet_vectors()

Stream vectors from Parquet files.

```python
def iter_parquet_vectors(
    file_path: str,
    index_column: str,
    vector_column: str,
    batch_size: int = 10000
) -> Iterator[Tuple[List[int], np.ndarray]]
```

**Yields:**
- Tuples of (indices, vectors) batches

## Configuration Classes

### HashSignatures

Container for LSH band signatures.

```python
@dataclass
class HashSignatures:
    bands: List[bytes]
```

## Examples

### Basic Usage

```python
import numpy as np
from lshrs import LSHRS

# Initialize with automatic configuration
lsh = LSHRS(dim=768, similarity_threshold=0.7)

# Index some vectors
for i in range(100):
    vec = np.random.randn(768).astype(np.float32)
    lsh.ingest(i, vec)

# Query for similar vectors
query = np.random.randn(768).astype(np.float32)
results = lsh.get_top_k(query, topk=10)
print(f"Top 10 similar indices: {results}")
```

### Advanced Usage with Reranking

```python
def fetch_vectors(indices):
    # Your custom vector retrieval logic
    return load_vectors_from_database(indices)

lsh = LSHRS(
    dim=768,
    num_bands=20,
    rows_per_band=10,
    vector_fetch_fn=fetch_vectors
)

# Query with cosine similarity reranking
query = np.random.randn(768).astype(np.float32)
scores = lsh.get_above_p(query, p=0.1)

for idx, similarity in scores:
    print(f"Index {idx}: similarity {similarity:.3f}")
```

### Persistence Example

```python
# Save the index
lsh.save_to_disk("my_index.lsh")

# Later, restore the index
restored_lsh = LSHRS.load_from_disk(
    "my_index.lsh",
    redis_host="production-redis"
)
```

## Performance Considerations

1. **Batch Size**: Use larger batch sizes (1k-100k) for better throughput
2. **Band/Row Balance**: More bands = higher recall; more rows per band = higher precision
3. **Redis Pipelining**: Automatic batching improves write performance
4. **Vector Normalization**: Pre-normalize vectors for consistent cosine similarity
5. **Memory Usage**: Buffer size affects memory consumption vs. throughput trade-off

## Error Handling

The library uses minimal error handling, letting underlying errors bubble up:

- **ValueError**: Dimension mismatches, invalid parameters
- **RuntimeError**: Missing configuration (e.g., no vector_fetch_fn for reranking)
- **ImportError**: Missing optional dependencies (psycopg, pyarrow)
- **Redis errors**: Connection issues, authentication failures

## Thread Safety

LSHRS instances are not thread-safe. Use separate instances per thread or implement external synchronization.
