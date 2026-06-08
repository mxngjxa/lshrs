# LSHRS - Redis-backed Locality Sensitive Hashing Requirements

## Executive Summary
LSHRS is a Redis-backed Locality Sensitive Hashing (LSH) package that adds cosine approximate-nearest-neighbor search to an existing datastore (PostgreSQL/pgvector, Parquet, object store, etc.) **without copying vectors into a dedicated vector database.** Redis holds only a compact `bucket -> set-of-indices` index — roughly two orders of magnitude smaller than the vectors it indexes — while the heavy vector payloads stay in the system of record and are fetched on demand for optional cosine reranking.

It is not positioned to beat a dedicated vector DB or RediSearch HNSW on recall or latency at scale. Its advantages are orthogonal: no data migration, a tiny Redis footprint, cheap exact deletes, and bring-your-own-datastore integration.

## Core Objectives

### Primary Goals
1. **Augment, Don't Replace, Your Datastore**: Add cosine ANN over the data you already store by keeping only a tiny LSH index in Redis and leaving vectors in the existing database; not a drop-in replacement for a dedicated vector DB at peak recall/QPS
2. **Integration-First Design**: Seamlessly work with existing database infrastructure without requiring data migration
3. **High Performance**: Support query latency < 100ms and indexing throughput > 10k vectors/sec on consumer hardware
4. **Simplicity**: Minimal, clean API with sensible defaults and minimal configuration overhead

### Non-Goals
- Multi-tenancy support (single Redis instance per database)
- Complex vector database features (metadata filtering, hybrid search)
- Real-time high-frequency updates (optimized for batch + occasional updates)

## Functional Requirements

### Data Model
1. **Vector Storage**: External database stores vectors in `index-vector_representation-raw_text` format
2. **Redis Storage**: Only stores `bucket→list-of-indices` mappings
3. **Vector Dimensions**: Flexible dimensionality support, fixed after initialization
4. **Index Management**: Support for batch indexing, single insertions, and deletions

### API Requirements

#### Core Interface
```python
class LSHRS:
    def __init__(self, **kwargs)  # Highly configurable with defaults
    def ingest(self, index: int, vector: np.ndarray)  # Single vector insertion
    def index(self, indices: List[int], vectors: Optional[np.ndarray])  # Batch indexing
    def query(self, vector: np.ndarray, top_k: Optional[int], top_p: Optional[float])  # Search
    def delete(self, indices: Union[int, List[int]])  # Remove vectors
```

#### Configuration Philosophy
- Extensive use of default parameters (similar to HuggingFace Transformers API)
- Minimal required configuration
- Auto-optimization of LSH parameters when not specified

### LSH Algorithm Requirements

#### Parameter Optimization
- Auto-compute optimal `num_bands` (b) and `rows_per_band` (r) using:
  - Square root heuristic: `b = sqrt(num_permutations)`, `r = num_permutations // b`
  - OR optimal false positive/negative rate calculation based on similarity threshold (t0 = 0.5)

#### Hashing Strategy
- Random projection LSH for cosine similarity
- Configurable number of hash functions (`num_perm`, default: 128)
- Reproducible hashing via seed parameter

### Query Processing

#### Retrieval Modes
1. **Top-K Only**: Fast candidate retrieval without reranking
2. **Top-P with Reranking**: Brute-force cosine similarity for top percentage
   - Requires `vector_fetch_fn` for external vector retrieval
   - Returns (index, score) tuples

#### Candidate Generation
- Union of all matching buckets across bands
- No duplicate filtering in bucket retrieval (handled by set operations)

## Non-Functional Requirements

### Performance Targets

> [!NOTE]
> These are **design targets, not measured benchmarks.** See [Performance Validation](#performance-validation) — no benchmark currently ships, so none of the figures below have been validated.

| Metric | Target | Conditions |
|--------|--------|------------|
| Query Latency (p95) | < 100ms | 6.4M vectors, consumer laptop |
| Indexing Throughput | > 10k vectors/sec | Batch mode, consumer laptop |
| Memory Usage | < 2GB | For hash tables and Redis operations |
| Insertion Rate | 100 vectors/hour | Single insertion mode |

### Scalability
- **Dataset Size**: Support 6.4M vectors (English Wikipedia)
- **Batch Size**: Configurable buffer size (default: 10,000)
- **Redis Operations**: Use pipelining for batch operations

### Persistence & Recovery
1. **State Serialization**: Save/load LSH configuration and random projection matrices
2. **Compact Storage**: Store only dimension + seed for ultra-compact persistence
3. **Reproducibility**: Identical results given same seed and configuration

### Integration Requirements

#### Database Support
- PostgreSQL (primary target)
- Any database with index-vector format
- Custom `vector_fetch_fn` for retrieval

#### Redis Configuration
- Single Redis instance per deployment
- Configurable connection parameters (host, port, db, password)
- Support for Redis Cluster (user-configured)

### Error Handling
- Minimal custom exceptions
- Let underlying errors (Redis, NumPy) bubble up
- Host application responsible for logging and monitoring

## Technical Architecture

### Package Structure
```
lshrs/
├── __init__.py          # Public exports (LSHRS, etc.)
├── core/
│   ├── __init__.py
│   └── main.py          # Main LSHRS orchestration class
├── hash/
│   ├── __init__.py
│   └── lsh.py           # Random-projection LSH hashing implementation
├── storage/
│   ├── __init__.py
│   └── redis.py         # Redis backend (bucket membership)
├── io/
│   ├── __init__.py
│   ├── postgres.py      # PostgreSQL streaming loader
│   └── parquet.py       # Parquet streaming loader
├── utils/
│   ├── __init__.py
│   ├── similarity.py    # Cosine similarity utilities
│   ├── norm.py          # Vector normalization helpers
│   └── br.py            # Band/row optimization + collision probability
└── _config/
    ├── __init__.py
    └── config.py        # Configuration defaults
```

> [!NOTE]
> Earlier drafts of this document described a flat `hasher.py` / `storage.py` / `similarity.py` layout. The shipping package uses the nested-package layout shown above; modules are organized under `core/`, `hash/`, `storage/`, `io/`, `utils/`, and `_config/`.

### Dependencies
- **Core**: numpy, redis-py
- **Optional**: scipy (for optimization algorithms)
- **Development**: pytest, pytest-redis

### Storage Schema

#### Redis Key Structure
```
lsh:{band_id}:bucket:{hash_hex} -> SET of indices
lsh:config -> Hash of configuration
lsh:state -> Serialized hasher state
```

## Testing Requirements

### Test Coverage
1. **Unit Tests**: All core functionality with mocked Redis
2. **Integration Tests**: Real Redis instance with sample data
3. **Performance Benchmarks**: Query and indexing benchmarks
4. **Demo Dataset**: Small parquet file for testing

### Test Data
- Sample parquet file with vectors
- Small Redis test instance
- Synthetic vector generation for edge cases

## Development Constraints

### Code Style
- Clean, minimal codebase
- No complex inheritance hierarchies
- Functional programming where appropriate
- Type hints for all public APIs

### Documentation
- Comprehensive docstrings
- Usage examples in README
- Performance tuning guide

## Success Criteria

### MVP Features
- [x] Top-K and Top-P retrieval
- [x] PostgreSQL integration example
- [x] State persistence
- [ ] **TODO (aspirational):** Validated batch indexing of 6.4M vectors — no benchmark currently ships, so this target is unverified.
- [ ] **TODO (aspirational):** Verified query latency < 100ms — no benchmark currently ships, so this target is unverified.

### Performance Validation

> [!IMPORTANT]
> The performance items below are **aspirational targets, not measured results.** No benchmark currently ships with the package, and the Wikipedia-scale validation has not yet been performed. Treat the numbers in the [Performance Targets](#performance-targets) table as design goals until a benchmark suite lands.

- [ ] **TODO:** Benchmark on the Wikipedia dataset
- [ ] **TODO:** Comparison with dedicated vector databases
- [ ] **TODO:** Memory and CPU profiling

## Future Considerations (Out of Scope for v1)

1. **Multi-tenancy**: Multiple indices with prefixes
2. **Dynamic rebalancing**: Adjust bands/rows on the fly
3. **Distributed processing**: Multi-node Redis cluster optimization
4. **Advanced similarity metrics**: Beyond cosine similarity
5. **Incremental indexing optimization**: Faster than linear deletion

## Appendix: Configuration Defaults

```python
DEFAULT_CONFIG = {
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'num_perm': 128,
    'num_bands': None,  # Auto-optimize
    'rows_per_band': None,  # Auto-optimize
    'buffer_size': 10000,  # NOTE: constructor default is 10,000 (not 8912)
    'seed': 42,
}
```

## References
- Datasketch library architecture
- MinHash LSH papers
- Redis best practices for set operations
