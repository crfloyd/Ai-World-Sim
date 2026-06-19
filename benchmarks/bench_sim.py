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
    python -m benchmarks.bench_sim --output results.jsonl

Output flags
------------
    --output FILE   append a JSON-lines record to FILE for tracking over time
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.observations import build_observation
from ai_world_sim.world.sim import WorldSim

from benchmarks._utils import BenchResult, _timed_loop, print_header, print_result


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
    benchmark reflects steady-state encoding, not cold-start memory.
    """
    sim = WorldSim(config=cfg, seed=seed)
    sim.generate()
    agent = sim.spawn_agents(1)[0]

    mem_cfg = cfg.get("memory", {})
    window = int(2 * mem_cfg.get("sight_radius", 10) + 1)
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
    """Benchmark WorldEnv.step() with REST — full Gymnasium round-trip.

    Uses REST (action 8) which is always valid, keeping the benchmark
    focused on step throughput rather than action dispatch branches.
    Episodes are reset automatically on termination.
    """
    env = WorldEnv(env_config={
        "world_config_override": {
            "world": cfg.get("world", {}),
            "animals": cfg.get("animals", {}),
        },
        "max_steps_per_episode": 10_000_000,
        "seed_range": [seed, seed],
    })
    env.reset(seed=seed)
    rest_action = 8  # REST — always valid

    def _step() -> None:
        _, _, terminated, truncated, _ = env.step(rest_action)
        if terminated or truncated:
            env.reset(seed=seed)

    _timed_loop(_step, warmup)
    return BenchResult.from_latencies("WorldEnv.step() [REST]", _timed_loop(_step, duration))


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

    print(f"\nAI World Sim — sim/obs/env benchmarks")
    print(f"  Python {sys.version.split()[0]}  |  {platform.machine()}")
    print(
        f"  world: {args.world_size}×{args.world_size}  seed: {args.seed}  "
        f"warmup: {args.warmup}s  duration: {args.duration}s  "
        f"animals: {'off' if args.no_animals else 'on'}"
    )
    print()
    print_header()

    results: list[BenchResult] = []

    for r in [
        bench_sim_tick(cfg, args.seed, args.warmup, args.duration),
        bench_observation_encoding(cfg, args.seed, args.warmup, args.duration),
        bench_env_step(cfg, args.seed, args.warmup, args.duration),
    ]:
        print_result(r)
        results.append(r)

    print()

    if args.output:
        record = {
            "benchmark": "bench_sim",
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
        with output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  Results appended to {output_path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI World Sim sim/obs/env benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds to run each benchmark after warmup.")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="Warmup seconds (excluded from timing).")
    parser.add_argument("--world-size", type=int, default=64,
                        help="World width and height in tiles.")
    parser.add_argument("--seed", type=int, default=42,
                        help="World generation seed.")
    parser.add_argument("--no-animals", action="store_true",
                        help="Disable scripted animals to isolate core sim overhead.")
    parser.add_argument("--output", metavar="FILE",
                        help="Append a JSON-lines record to FILE for tracking over time.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
