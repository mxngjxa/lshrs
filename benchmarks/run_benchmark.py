#!/usr/bin/env python3
"""
LSHRS benchmark harness.

Turns the library's positioning claims into reproducible numbers on synthetic
data. Measures, for configurable N / dim / num_perm / threshold:

1. Index build throughput (vectors/sec) via batched ``index()``.
2. Query latency p50/p95 for top-k (no rerank) and top-p (with an in-memory
   ``vector_fetch_fn`` standing in for the external datastore).
3. Redis footprint asymmetry: bytes used by the bucket index vs bytes the raw
   vectors would occupy (the core "~100x smaller" claim).
4. Delete latency: time to delete K random indices (exercises the reverse
   index / fast delete path).

Redis backend
-------------
By default the harness runs against an in-memory ``fakeredis`` server so it
works anywhere with no external services. ``fakeredis`` gives *functional*
timings but does NOT implement the ``MEMORY USAGE`` / ``INFO memory`` commands,
so footprint on fakeredis is *estimated* analytically (number of SADD members x
a per-member cost) rather than measured. Pass ``--real-redis`` (or set
``LSHRS_BENCH_REAL_REDIS=1``) to point at a real Redis server and get true
measured memory.

This file is a runnable script, NOT part of the pytest suite. A tiny smoke test
(``tests/test_benchmark_smoke.py``) imports and runs it with a small N to
prevent bitrot.

Usage
-----
    uv run python benchmarks/run_benchmark.py --n 5000 --dim 256 --num-perm 128
    uv run python benchmarks/run_benchmark.py --real-redis --redis-host localhost
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lshrs import LSHRS
from lshrs.storage.redis import RedisStorage

# Default location of the results template written next to this script.
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS_TEMPLATE.md")


# ---------------------------------------------------------------------------
# Data + storage helpers
# ---------------------------------------------------------------------------


def make_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate ``n`` random unit vectors of dimension ``dim`` (float32)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def build_storage(
    *,
    real_redis: bool,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    prefix: str,
) -> RedisStorage:
    """
    Construct a RedisStorage, swapping in a fakeredis client unless ``real_redis``.

    RedisStorage builds its own connection pool + client in ``__init__``. For the
    fakeredis path we build the storage object (the pool is harmless/unused) and
    then replace ``_client`` with an in-memory FakeStrictRedis instance.
    """
    storage = RedisStorage(host=redis_host, port=redis_port, db=redis_db, prefix=prefix)

    if not real_redis:
        import fakeredis

        storage._client = fakeredis.FakeStrictRedis()

    # Clear after wiring the client so we start from a clean namespace
    # (no-op on a fresh fakeredis; removes leftover keys on real Redis).
    storage.clear()
    return storage


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _percentile(samples_ms: list[float], pct: float) -> float:
    if not samples_ms:
        return float("nan")
    ordered = sorted(samples_ms)
    # Nearest-rank percentile.
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    count: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        if not samples_ms:
            return cls(float("nan"), float("nan"), float("nan"), 0)
        return cls(
            p50_ms=_percentile(samples_ms, 50),
            p95_ms=_percentile(samples_ms, 95),
            mean_ms=statistics.fmean(samples_ms),
            count=len(samples_ms),
        )


@dataclass
class BenchmarkResults:
    n: int
    dim: int
    num_perm: int
    threshold: float
    num_bands: int
    rows_per_band: int
    backend: str

    build_seconds: float = 0.0
    build_throughput: float = 0.0

    topk_latency: Optional[LatencyStats] = None
    topp_latency: Optional[LatencyStats] = None

    redis_index_bytes: int = 0
    redis_bytes_measured: bool = False
    raw_vector_bytes: int = 0
    footprint_ratio: float = 0.0

    delete_count: int = 0
    delete_seconds: float = 0.0

    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Footprint measurement
# ---------------------------------------------------------------------------


def measure_redis_index_bytes(storage: RedisStorage) -> tuple[int, bool]:
    """
    Return ``(bytes, measured)`` for the bucket index under the storage prefix.

    On real Redis we sum ``MEMORY USAGE`` over every bucket key (true resident
    bytes incl. set/encoding overhead). On fakeredis those commands are absent,
    so we *estimate*: count total SADD members across all buckets and multiply
    by a conservative per-member cost, plus a flat per-key overhead.
    """
    client = storage._client
    pattern = f"{storage.prefix}:*:bucket:*"
    keys = list(client.scan_iter(match=pattern))

    # Try the real measurement path first.
    try:
        total = 0
        for key in keys:
            usage = client.memory_usage(key)
            if usage is not None:
                total += int(usage)
        return total, True
    except Exception:
        pass

    # Estimation fallback (fakeredis). Redis intset/hashtable members cost a few
    # bytes plus per-key dict + object overhead. These are rough constants.
    PER_MEMBER_BYTES = 16
    PER_KEY_OVERHEAD = 64
    total = PER_KEY_OVERHEAD * len(keys)
    with storage.pipeline() as pipe:
        for key in keys:
            pipe.scard(key)
        cards = pipe.execute()
    total += PER_MEMBER_BYTES * int(sum(cards))
    return total, False


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------


def bench_build(lsh: LSHRS, indices: list[int], vectors: np.ndarray, batch_size: int) -> tuple[float, float]:
    """Index all vectors in batches; return (seconds, vectors/sec)."""
    start = time.perf_counter()
    for begin in range(0, len(indices), batch_size):
        chunk = slice(begin, begin + batch_size)
        lsh.index(indices[chunk], vectors[chunk])
    elapsed = time.perf_counter() - start
    throughput = len(indices) / elapsed if elapsed > 0 else float("inf")
    return elapsed, throughput


def bench_query(
    lsh: LSHRS,
    queries: np.ndarray,
    *,
    top_k: int,
    top_p: Optional[float],
) -> LatencyStats:
    """Time per-query latency over a set of query vectors."""
    samples_ms: list[float] = []
    for q in queries:
        start = time.perf_counter()
        if top_p is None:
            lsh.query(q, top_k=top_k, top_p=None)
        else:
            lsh.query(q, top_k=top_k, top_p=top_p)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return LatencyStats.from_samples(samples_ms)


def bench_delete(lsh: LSHRS, indices: list[int], k: int, seed: int = 7) -> tuple[int, float]:
    """Delete ``k`` random indices in one call; return (count, seconds)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(indices))
    victims = rng.choice(np.asarray(indices), size=k, replace=False).tolist()
    start = time.perf_counter()
    lsh.delete(victims)
    return k, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    n: int = 5000,
    dim: int = 256,
    num_perm: int = 128,
    threshold: float = 0.5,
    batch_size: int = 1000,
    num_queries: int = 200,
    top_k: int = 10,
    top_p: float = 0.1,
    delete_k: int = 100,
    seed: int = 0,
    real_redis: bool = False,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    prefix: str = "lsh_bench",
    write_results: bool = True,
    results_path: str = RESULTS_PATH,
    quiet: bool = False,
) -> BenchmarkResults:
    """Run the full benchmark suite and (optionally) write a markdown report."""

    vectors = make_unit_vectors(n, dim, seed=seed)
    indices = list(range(n))
    store: dict[int, np.ndarray] = dict(zip(indices, vectors, strict=True))

    def vector_fetch_fn(idxs: list[int]) -> np.ndarray:
        # Stand-in for the external datastore: an in-memory dict lookup.
        return np.stack([store[int(i)] for i in idxs])

    storage = build_storage(
        real_redis=real_redis,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        prefix=prefix,
    )

    lsh = LSHRS(
        dim=dim,
        num_perm=num_perm,
        similarity_threshold=threshold,
        storage=storage,
        vector_fetch_fn=vector_fetch_fn,
        seed=seed,
    )
    stats = lsh.stats()

    results = BenchmarkResults(
        n=n,
        dim=dim,
        num_perm=num_perm,
        threshold=threshold,
        num_bands=stats["num_bands"],
        rows_per_band=stats["rows_per_band"],
        backend="real-redis" if real_redis else "fakeredis",
    )

    # 1. Build throughput.
    results.build_seconds, results.build_throughput = bench_build(lsh, indices, vectors, batch_size)

    # 2. Query latency (top-k and top-p).
    q_count = min(num_queries, n)
    queries = vectors[:q_count]
    results.topk_latency = bench_query(lsh, queries, top_k=top_k, top_p=None)
    results.topp_latency = bench_query(lsh, queries, top_k=top_k, top_p=top_p)

    # 3. Footprint asymmetry.
    results.redis_index_bytes, results.redis_bytes_measured = measure_redis_index_bytes(storage)
    results.raw_vector_bytes = n * dim * 4  # float32
    if results.redis_index_bytes > 0:
        results.footprint_ratio = results.raw_vector_bytes / results.redis_index_bytes

    # 4. Delete latency.
    results.delete_count, results.delete_seconds = bench_delete(lsh, indices, delete_k, seed=seed + 1)

    if not results.redis_bytes_measured:
        results.notes.append(
            "Redis footprint is ESTIMATED (fakeredis lacks MEMORY USAGE/INFO). Use --real-redis for measured bytes."
        )
    results.notes.append("All numbers are machine-dependent and meant as relative, reproducible signals.")

    lsh.clear()
    lsh.close()

    table = render_markdown(results)
    if not quiet:
        print(table)
    if write_results:
        with open(results_path, "w") as f:
            f.write(table)
        if not quiet:
            print(f"\nWrote results to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def render_markdown(r: BenchmarkResults) -> str:
    topk = r.topk_latency or LatencyStats.from_samples([])
    topp = r.topp_latency or LatencyStats.from_samples([])
    measured = "measured" if r.redis_bytes_measured else "estimated"
    del_per = (r.delete_seconds / r.delete_count * 1000.0) if r.delete_count else float("nan")

    lines = [
        "# LSHRS Benchmark Results",
        "",
        "> Numbers below are **machine-dependent** and were produced by "
        "`benchmarks/run_benchmark.py` on synthetic random unit vectors. "
        "Treat them as reproducible *relative* signals, not absolute guarantees.",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Vectors (N) | {r.n:,} |",
        f"| Dimension | {r.dim} |",
        f"| num_perm | {r.num_perm} |",
        f"| similarity_threshold | {r.threshold} |",
        f"| num_bands x rows_per_band | {r.num_bands} x {r.rows_per_band} |",
        f"| Redis backend | {r.backend} |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| **Build throughput** | {_fmt(r.build_throughput)} vectors/sec ({_fmt(r.build_seconds)} s total) |",
        f"| **Top-k latency** (no rerank) p50 / p95 | {_fmt(topk.p50_ms, 3)} ms / {_fmt(topk.p95_ms, 3)} ms |",
        f"| **Top-p latency** (cosine rerank) p50 / p95 | {_fmt(topp.p50_ms, 3)} ms / {_fmt(topp.p95_ms, 3)} ms |",
        f"| **Raw vector bytes** (N x dim x 4) | {r.raw_vector_bytes:,} B ({_fmt(r.raw_vector_bytes / 1e6)} MB) |",
        f"| **Redis index bytes** ({measured}) | {r.redis_index_bytes:,} B ({_fmt(r.redis_index_bytes / 1e6)} MB) |",
        f"| **Footprint ratio** (raw / index) | **{_fmt(r.footprint_ratio)}x smaller** |",
        f"| **Delete latency** ({r.delete_count} indices) | "
        f"{_fmt(r.delete_seconds * 1000.0)} ms total ({_fmt(del_per, 4)} ms/index) |",
        "",
    ]
    if r.notes:
        lines.append("## Notes")
        lines.append("")
        for note in r.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000, help="Number of synthetic vectors to index.")
    p.add_argument("--dim", type=int, default=256, help="Vector dimensionality.")
    p.add_argument("--num-perm", type=int, default=128, help="Total LSH projection bits.")
    p.add_argument("--threshold", type=float, default=0.5, help="Target similarity threshold (auto bands/rows).")
    p.add_argument("--batch-size", type=int, default=1000, help="Index batch size.")
    p.add_argument("--num-queries", type=int, default=200, help="Number of query vectors to time.")
    p.add_argument("--top-k", type=int, default=10, help="top_k for queries.")
    p.add_argument("--top-p", type=float, default=0.1, help="top_p for rerank queries.")
    p.add_argument("--delete-k", type=int, default=100, help="Number of random indices to delete.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--real-redis",
        action="store_true",
        default=_env_flag("LSHRS_BENCH_REAL_REDIS"),
        help="Use a real Redis server (enables measured footprint). Also enabled via LSHRS_BENCH_REAL_REDIS=1.",
    )
    p.add_argument("--redis-host", default=os.environ.get("LSHRS_BENCH_REDIS_HOST", "localhost"))
    p.add_argument("--redis-port", type=int, default=int(os.environ.get("LSHRS_BENCH_REDIS_PORT", "6379")))
    p.add_argument("--redis-db", type=int, default=int(os.environ.get("LSHRS_BENCH_REDIS_DB", "0")))
    p.add_argument("--prefix", default="lsh_bench", help="Redis key prefix for the benchmark namespace.")
    p.add_argument("--no-write", action="store_true", help="Do not write RESULTS_TEMPLATE.md.")
    return p


def main(argv: Optional[list[str]] = None) -> BenchmarkResults:
    args = build_arg_parser().parse_args(argv)
    return run_benchmark(
        n=args.n,
        dim=args.dim,
        num_perm=args.num_perm,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_queries=args.num_queries,
        top_k=args.top_k,
        top_p=args.top_p,
        delete_k=args.delete_k,
        seed=args.seed,
        real_redis=args.real_redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        prefix=args.prefix,
        write_results=not args.no_write,
    )


if __name__ == "__main__":
    main()
