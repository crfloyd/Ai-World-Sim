"""Micro-benchmarks for the three hot paths in AI World Sim.

Measures
--------
1. WorldSim tick rate          raw simulation throughput (ticks/s)
2. Observation encoding rate   build_observation() throughput (obs/s)
3. Gym env step rate           WorldEnv.step() end-to-end throughput (steps/s)

Each benchmark runs a warmup phase (excluded from timing) followed by a
timed phase.  Per-call latencies are collected and summarised as throughput,
mean, median, and standard deviation.

Usage
-----
    python -m benchmarks.bench_sim
    python -m benchmarks.bench_sim --duration 10 --warmup 2
    python -m benchmarks.bench_sim --world-size 32 --no-animals
    python -m benchmarks.bench_sim --output results.json

Output flags
------------
    --output FILE   write a JSON record to FILE for tracking over time
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.observations import build_observation
from ai_world_sim.world.sim import WorldSim


# ---------------------------------------------------------------------------
# Core timing primitives
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_sim_tick(
    cfg: dict,
    seed: int,
    warmup: float,
    duration: float,
) -> BenchResult:
    """Benchmark WorldSim.tick() — pure simulation overhead."""
    sim = WorldSim(config=cfg, seed=seed)
    sim.generate()
    sim.spawn_agents(1)

    _timed_loop(sim.tick, warmup)
    return BenchResult.from_latencies("WorldSim.tick()", _timed_loop(sim.tick, duration))


def bench_observation_encoding(
    cfg: dict,
    seed: int,
    warmup: float,
    duration: float,
) -> BenchResult:
    """Benchmark build_observation() — observation encoding overhead.

    Memory is primed once via update_agent_memory() before timing so the
    benchmark reflects steady-state observation building, not cold-start memory.
    """
    sim = WorldSim(config=cfg, seed=seed)
    sim.generate()
    agent = sim.spawn_agents(1)[0]

    mem_cfg = cfg.get("memory", {})
    window = int(2 * mem_cfg.get("sight_radius", 10) + 1)

    # Prime memory so summarize() has representative data to encode.
    sim.update_agent_memory(agent)

    fn = lambda: build_observation(sim, agent, window=window)
    _timed_loop(fn, warmup)
    return BenchResult.from_latencies("build_observation()", _timed_loop(fn, duration))


def bench_env_step(
    cfg: dict,
    seed: int,
    warmup: float,
    duration: float,
) -> BenchResult:
    """Benchmark WorldEnv.step() — full Gymnasium round-trip.

    Uses REST (action 8) which is always valid, avoiding mask violations
    and keeping the benchmark focused on step throughput rather than action
    dispatch branches.  Episodes are reset automatically on termination.
    """
    env = WorldEnv(env_config={
        "world_config_override": {
            "world": cfg.get("world", {}),
            "animals": cfg.get("animals", {}),
        },
        "max_steps_per_episode": 10_000_000,   # don't let truncation break the loop
        "seed_range": [seed, seed],
    })
    env.reset(seed=seed)
    rest_action = 8  # REST — always valid

    def _step() -> None:
        _, _, terminated, truncated, _ = env.step(rest_action)
        if terminated or truncated:
            env.reset(seed=seed)

    _timed_loop(_step, warmup)
    return BenchResult.from_latencies("WorldEnv.step()", _timed_loop(_step, duration))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_W = 24

def _print_header() -> None:
    print(
        f"  {'Benchmark':<{_COL_W}}  {'Throughput':>12}   "
        f"{'Mean':>9}  {'Median':>9}  {'p99':>9}  {'σ':>7}  n"
    )
    print(
        f"  {'-'*_COL_W}  {'-'*12}   "
        f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  ---"
    )


def _print_result(r: BenchResult) -> None:
    print(
        f"  {r.label:<{_COL_W}}  {r.throughput:>10,.0f}/s   "
        f"{r.mean_us:>7.1f}µs  {r.median_us:>7.1f}µs  "
        f"{r.p99_us:>7.1f}µs  {r.stdev_us:>5.1f}µs  {r.n}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = args.world_size
    cfg["world"]["height"] = args.world_size
    if args.no_animals:
        cfg["animals"]["wolves_per_world"] = 0
        cfg["animals"]["rabbits_per_world"] = 0
        cfg["animals"]["deer_per_world"] = 0

    print(f"\nAI World Sim — performance benchmarks")
    print(f"  Python {sys.version.split()[0]}  |  {platform.machine()}")
    print(
        f"  world: {args.world_size}×{args.world_size}  seed: {args.seed}  "
        f"warmup: {args.warmup}s  duration: {args.duration}s  "
        f"animals: {'off' if args.no_animals else 'on'}"
    )
    print()
    _print_header()

    results: list[BenchResult] = []

    r = bench_sim_tick(cfg, args.seed, args.warmup, args.duration)
    _print_result(r)
    results.append(r)

    r = bench_observation_encoding(cfg, args.seed, args.warmup, args.duration)
    _print_result(r)
    results.append(r)

    r = bench_env_step(cfg, args.seed, args.warmup, args.duration)
    _print_result(r)
    results.append(r)

    print()

    if args.output:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.machine(),
            "world_size": args.world_size,
            "seed": args.seed,
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "no_animals": args.no_animals,
            "results": [asdict(r) for r in results],
        }
        output_path = Path(args.output)
        # Append to a JSON-lines file if it already exists, otherwise create it.
        with output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  Results appended to {output_path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI World Sim performance benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Seconds to run each benchmark after warmup.",
    )
    parser.add_argument(
        "--warmup", type=float, default=1.0,
        help="Warmup seconds (excluded from timing).",
    )
    parser.add_argument(
        "--world-size", type=int, default=64,
        help="World width and height in tiles.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="World generation seed.",
    )
    parser.add_argument(
        "--no-animals", action="store_true",
        help="Disable scripted animals to isolate core sim overhead.",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Append a JSON record to FILE for tracking results over time.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
