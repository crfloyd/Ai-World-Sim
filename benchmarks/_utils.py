"""Shared timing primitives for AI World Sim benchmarks."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable


_COL_W = 24


def _timed_loop(fn: Callable[[], None], duration: float) -> list[float]:
    """Call *fn* in a tight loop for *duration* seconds.

    Returns a list of per-call wall-clock latencies in seconds.
    """
    latencies: list[float] = []
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - t0)
    return latencies


@dataclass
class BenchResult:
    label: str
    n: int
    throughput: float       # calls/second
    mean_us: float          # microseconds
    median_us: float
    stdev_us: float
    p99_us: float

    @classmethod
    def from_latencies(cls, label: str, latencies: list[float]) -> "BenchResult":
        n = len(latencies)
        mean = statistics.mean(latencies)
        sorted_lat = sorted(latencies)
        p99_idx = max(0, int(0.99 * n) - 1)
        return cls(
            label=label,
            n=n,
            throughput=1.0 / mean,
            mean_us=mean * 1e6,
            median_us=statistics.median(latencies) * 1e6,
            stdev_us=statistics.stdev(latencies) * 1e6 if n > 1 else 0.0,
            p99_us=sorted_lat[p99_idx] * 1e6,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def print_header() -> None:
    print(
        f"  {'Benchmark':<{_COL_W}}  {'Throughput':>12}   "
        f"{'Mean':>9}  {'Median':>9}  {'p99':>9}  {'σ':>7}  n"
    )
    print(
        f"  {'-'*_COL_W}  {'-'*12}   "
        f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  ---"
    )


def print_result(r: BenchResult) -> None:
    print(
        f"  {r.label:<{_COL_W}}  {r.throughput:>10,.0f}/s   "
        f"{r.mean_us:>7.1f}µs  {r.median_us:>7.1f}µs  "
        f"{r.p99_us:>7.1f}µs  {r.stdev_us:>5.1f}µs  {r.n}"
    )
