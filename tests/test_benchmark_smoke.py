"""Smoke test: import the benchmark harness and run it small under fakeredis.

This keeps ``benchmarks/run_benchmark.py`` from bitrotting. It is intentionally
tiny (N=200) and never writes the results file. The benchmark itself is a
script, not part of the measured suite.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

pytest.importorskip("fakeredis")

_BENCH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "run_benchmark.py",
)


def _load_harness():
    module_name = "lshrs_run_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses with PEP 563 annotations can resolve
    # the module's namespace via sys.modules during class creation.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_runs_small_under_fakeredis() -> None:
    harness = _load_harness()
    results = harness.run_benchmark(
        n=200,
        dim=32,
        num_perm=16,
        batch_size=50,
        num_queries=10,
        delete_k=5,
        real_redis=False,
        write_results=False,
        quiet=True,
        prefix="lsh_bench_smoke",
    )

    assert results.n == 200
    assert results.backend == "fakeredis"
    assert results.build_throughput > 0
    assert results.topk_latency is not None and results.topk_latency.count == 10
    assert results.topp_latency is not None and results.topp_latency.count == 10
    assert results.raw_vector_bytes == 200 * 32 * 4
    assert results.redis_index_bytes > 0
    assert results.footprint_ratio > 0
    assert results.redis_bytes_measured is False  # fakeredis cannot measure
    assert results.delete_count == 5


def test_markdown_render_contains_key_sections() -> None:
    harness = _load_harness()
    results = harness.run_benchmark(
        n=200,
        dim=32,
        num_perm=16,
        batch_size=50,
        num_queries=5,
        delete_k=5,
        real_redis=False,
        write_results=False,
        quiet=True,
        prefix="lsh_bench_smoke2",
    )
    table = harness.render_markdown(results)
    assert "Build throughput" in table
    assert "Footprint ratio" in table
    assert "Delete latency" in table
    assert "machine-dependent" in table
