# LSHRS Benchmarks

A runnable harness that turns LSHRS's positioning claims into reproducible
numbers on **synthetic data** (random unit vectors).

## What it measures

1. **Index build throughput** — vectors/sec via batched `LSHRS.index()`.
2. **Query latency** — p50/p95 for top-k (no rerank) and top-p (cosine rerank
   with an in-memory `vector_fetch_fn` standing in for the external datastore).
3. **Redis footprint asymmetry** — bytes used by the bucket index vs the bytes
   the raw vectors would occupy (`N x dim x 4`, float32). Reports the ratio
   behind the "~100x smaller" claim. The realized ratio scales roughly with
   `dim x 4 / (num_bands x per_member_cost)`, so higher-dimensional embeddings
   with fewer/wider bands show the largest asymmetry.
4. **Delete latency** — time to delete K random indices via `LSHRS.delete()`
   (exercises `RedisStorage.remove_indices`, which scans every bucket key).

## Running

```bash
# Default: fakeredis, N=5000, dim=256, num_perm=128
uv run python benchmarks/run_benchmark.py

# Custom run
uv run python benchmarks/run_benchmark.py --n 50000 --dim 768 --num-perm 256 --threshold 0.7
```

Results are printed as a markdown table and written to
[`RESULTS_TEMPLATE.md`](RESULTS_TEMPLATE.md). Pass `--no-write` to skip the file.

### Redis backends

By default the harness uses an in-memory **`fakeredis`** server, so it runs
anywhere with no external services. `fakeredis` provides *functional* timings
but does **not** implement `MEMORY USAGE` / `INFO memory`, so the footprint on
fakeredis is **estimated** analytically (total SADD members x a per-member cost
+ per-key overhead) rather than measured.

For **true measured memory**, point at a real Redis server:

```bash
uv run python benchmarks/run_benchmark.py --real-redis --redis-host localhost --redis-port 6379
# or via env:
LSHRS_BENCH_REAL_REDIS=1 uv run python benchmarks/run_benchmark.py
```

On real Redis the harness sums `MEMORY USAGE` over every bucket key.

## Notes

- All numbers are **machine-dependent**. Treat them as reproducible *relative*
  signals, not absolute guarantees.
- The benchmark is a script, not part of the pytest suite. A tiny smoke test
  (`tests/test_benchmark_smoke.py`) runs it with N=200 under fakeredis to
  prevent bitrot.
