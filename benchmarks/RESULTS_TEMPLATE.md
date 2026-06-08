# LSHRS Benchmark Results

> Numbers below are **machine-dependent** and were produced by `benchmarks/run_benchmark.py` on synthetic random unit vectors. Treat them as reproducible *relative* signals, not absolute guarantees.

## Configuration

| Parameter | Value |
| --- | --- |
| Vectors (N) | 5,000 |
| Dimension | 256 |
| num_perm | 128 |
| similarity_threshold | 0.5 |
| num_bands x rows_per_band | 8 x 16 |
| Redis backend | fakeredis |

## Results

| Metric | Value |
| --- | --- |
| **Build throughput** | 7,637.56 vectors/sec (0.65 s total) |
| **Top-k latency** (no rerank) p50 / p95 | 0.151 ms / 0.174 ms |
| **Top-p latency** (cosine rerank) p50 / p95 | 0.177 ms / 0.222 ms |
| **Raw vector bytes** (N x dim x 4) | 5,120,000 B (5.12 MB) |
| **Redis index bytes** (estimated) | 3,087,104 B (3.09 MB) |
| **Footprint ratio** (raw / index) | **1.66x smaller** |
| **Delete latency** (100 indices) | 40,342.66 ms total (403.4266 ms/index) |

## Notes

- Redis footprint is ESTIMATED (fakeredis lacks MEMORY USAGE/INFO). Use --real-redis for measured bytes.
- All numbers are machine-dependent and meant as relative, reproducible signals.
