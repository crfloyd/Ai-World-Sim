"""Policy inference and rollout throughput benchmarks for AI World Sim.

Measures
--------
1. AgentBrain forward (batch=1)       single-sample inference latency
2. AgentBrain forward — batched       forward pass at batch sizes 1/32/128/512
3. WorldEnv.step + random valid       env step with action sampled from mask
4. WorldEnv.step + policy action      env step with action chosen by model
5. Policy rollout                     full collect loop — samples/s achievable
                                      by one in-process worker (no RLlib overhead)

Reading the numbers
-------------------
Compare (1) vs (3) to see model overhead vs env overhead.
Compare (3) vs (4) to see how much inference costs per step.
Compare (5) against (4) to see whether rollout collection is truly serial.
The gap between (5) and full RLlib throughput reflects Ray serialisation
and multi-worker coordination overhead (not measured here).

Usage
-----
    python -m benchmarks.bench_policy
    python -m benchmarks.bench_policy --duration 10 --batch-sizes 1 64 256 1024
    python -m benchmarks.bench_policy --device cuda          # if CUDA available
    python -m benchmarks.bench_policy --output results.jsonl

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

import numpy as np
import torch

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.model import AgentBrain
from ai_world_sim.rl.observations import NUM_ACTIONS, NUM_CHANNELS, SELF_DIM, build_observation
from ai_world_sim.world.memory import MEMORY_DIM

from benchmarks._utils import BenchResult, _timed_loop, print_header, print_result

_WINDOW = 21  # matches default sight_radius=10 → 2*10+1


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_env(cfg: dict, seed: int) -> WorldEnv:
    return WorldEnv(env_config={
        "world_config_override": {
            "world": cfg.get("world", {}),
            "animals": cfg.get("animals", {}),
        },
        "max_steps_per_episode": 10_000_000,
        "seed_range": [seed, seed],
    })


def _random_dummy_inputs(
    batch: int,
    window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-initialised tensors matching the policy input contract."""
    grid = torch.zeros(batch, NUM_CHANNELS, window, window, device=device)
    self_f = torch.zeros(batch, SELF_DIM, device=device)
    mem_f = torch.zeros(batch, MEMORY_DIM, device=device)
    mask = torch.ones(batch, NUM_ACTIONS, device=device)
    return grid, self_f, mem_f, mask


def _obs_to_tensors(
    obs: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a single Gymnasium observation dict to batched tensors (B=1)."""
    return (
        torch.from_numpy(obs["local_grid"]).unsqueeze(0).to(device),
        torch.from_numpy(obs["self_features"]).unsqueeze(0).to(device),
        torch.from_numpy(obs["memory_features"]).unsqueeze(0).to(device),
        torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(device),
    )


def _sample_valid_action(mask: np.ndarray) -> int:
    """Sample uniformly from actions where mask == 1."""
    valid = np.where(mask > 0.5)[0]
    return int(np.random.choice(valid))


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_brain_forward_single(
    brain: AgentBrain,
    device: torch.device,
    warmup: float,
    duration: float,
) -> BenchResult:
    """Single-sample forward pass (batch=1)."""
    grid, self_f, mem_f, mask = _random_dummy_inputs(1, _WINDOW, device)

    with torch.no_grad():
        fn = lambda: brain(grid, self_f, mem_f, mask)
        _timed_loop(fn, warmup)
        return BenchResult.from_latencies(
            "brain.forward() [B=1]", _timed_loop(fn, duration)
        )


def bench_brain_forward_batched(
    brain: AgentBrain,
    device: torch.device,
    batch_sizes: list[int],
    warmup: float,
    duration: float,
) -> list[BenchResult]:
    """Forward pass at multiple batch sizes. Reports per-sample throughput."""
    results: list[BenchResult] = []
    with torch.no_grad():
        for b in batch_sizes:
            grid, self_f, mem_f, mask = _random_dummy_inputs(b, _WINDOW, device)

            fn = lambda: brain(grid, self_f, mem_f, mask)  # noqa: B023
            _timed_loop(fn, warmup)
            raw = _timed_loop(fn, duration)

            # Convert per-batch latencies to per-sample latencies.
            per_sample = [lat / b for lat in raw]
            results.append(BenchResult.from_latencies(
                f"brain.forward() [B={b}]", per_sample
            ))
    return results


def bench_env_random_valid(
    cfg: dict,
    seed: int,
    warmup: float,
    duration: float,
) -> BenchResult:
    """WorldEnv.step() with action sampled uniformly from the valid mask.

    More representative of realistic training than always-REST, since it
    exercises all action dispatch branches in proportion to mask availability.
    """
    env = _make_env(cfg, seed)
    obs, _ = env.reset(seed=seed)

    def _step() -> None:
        nonlocal obs
        action = _sample_valid_action(obs["action_mask"])
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    _timed_loop(_step, warmup)
    return BenchResult.from_latencies(
        "WorldEnv.step() [rand-valid]", _timed_loop(_step, duration)
    )


def bench_env_policy_action(
    cfg: dict,
    seed: int,
    brain: AgentBrain,
    device: torch.device,
    warmup: float,
    duration: float,
) -> BenchResult:
    """WorldEnv.step() with action selected by the policy model.

    Measures the combined cost of observation encoding (already done by env),
    tensor conversion, model inference, and env stepping — the critical inner
    loop of a single-process PPO rollout worker.
    """
    env = _make_env(cfg, seed)
    obs, _ = env.reset(seed=seed)

    def _step() -> None:
        nonlocal obs
        grid, self_f, mem_f, mask = _obs_to_tensors(obs, device)
        with torch.no_grad():
            action, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=False)
        obs, _, terminated, truncated, _ = env.step(action.item())
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    _timed_loop(_step, warmup)
    return BenchResult.from_latencies(
        "WorldEnv.step() [policy]", _timed_loop(_step, duration)
    )


def bench_policy_rollout(
    cfg: dict,
    seed: int,
    brain: AgentBrain,
    device: torch.device,
    rollout_steps: int,
    n_rollouts: int,
    warmup: float,
) -> BenchResult:
    """In-process policy rollout — samples/s from one worker.

    Collects *rollout_steps* transitions in a tight loop identical to what
    a single RLlib rollout worker does, minus Ray serialisation overhead.
    Running *n_rollouts* consecutive rollouts gives the sustained throughput.

    This is the tightest lower-bound on RLlib single-worker throughput.
    Real RLlib throughput will be lower due to worker communication overhead.
    """
    env = _make_env(cfg, seed)
    obs, _ = env.reset(seed=seed)

    # Warmup
    warmup_env = _make_env(cfg, seed)
    warmup_obs, _ = warmup_env.reset(seed=seed)
    import time
    warmup_deadline = time.perf_counter() + warmup
    while time.perf_counter() < warmup_deadline:
        g, s, m, mk = _obs_to_tensors(warmup_obs, device)
        with torch.no_grad():
            act, _, _ = brain.act(g, s, m, mk, deterministic=False)
        warmup_obs, _, term, trunc, _ = warmup_env.step(act.item())
        if term or trunc:
            warmup_obs, _ = warmup_env.reset(seed=seed)

    # Timed rollouts
    import time as _time
    latencies: list[float] = []
    for _ in range(n_rollouts):
        t0 = _time.perf_counter()
        for _ in range(rollout_steps):
            grid, self_f, mem_f, mask = _obs_to_tensors(obs, device)
            with torch.no_grad():
                action, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                obs, _ = env.reset(seed=seed)
        elapsed = _time.perf_counter() - t0
        # Report per-sample latency so throughput = samples/second.
        latencies.extend([elapsed / rollout_steps] * rollout_steps)

    return BenchResult.from_latencies(
        f"policy rollout [{rollout_steps}×{n_rollouts}]", latencies
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    brain = AgentBrain().to(device)
    brain.eval()

    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = args.world_size
    cfg["world"]["height"] = args.world_size
    if args.no_animals:
        cfg["animals"]["wolves_per_world"] = 0
        cfg["animals"]["rabbits_per_world"] = 0
        cfg["animals"]["deer_per_world"] = 0

    batch_sizes: list[int] = args.batch_sizes

    print(f"\nAI World Sim — policy inference & rollout benchmarks")
    print(f"  Python {sys.version.split()[0]}  |  {platform.machine()}  |  device: {device}")
    print(
        f"  world: {args.world_size}×{args.world_size}  seed: {args.seed}  "
        f"warmup: {args.warmup}s  duration: {args.duration}s  "
        f"animals: {'off' if args.no_animals else 'on'}"
    )
    print(f"  batch sizes: {batch_sizes}  rollout: {args.rollout_steps}×{args.n_rollouts}")
    print()

    results: list[BenchResult] = []

    print_header()

    r = bench_brain_forward_single(brain, device, args.warmup, args.duration)
    print_result(r)
    results.append(r)

    for r in bench_brain_forward_batched(brain, device, batch_sizes, args.warmup, args.duration):
        print_result(r)
        results.append(r)

    r = bench_env_random_valid(cfg, args.seed, args.warmup, args.duration)
    print_result(r)
    results.append(r)

    r = bench_env_policy_action(cfg, args.seed, brain, device, args.warmup, args.duration)
    print_result(r)
    results.append(r)

    r = bench_policy_rollout(
        cfg, args.seed, brain, device,
        rollout_steps=args.rollout_steps,
        n_rollouts=args.n_rollouts,
        warmup=args.warmup,
    )
    print_result(r)
    results.append(r)

    print()

    if args.output:
        record = {
            "benchmark": "bench_policy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.machine(),
            "device": str(device),
            "world_size": args.world_size,
            "seed": args.seed,
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "no_animals": args.no_animals,
            "batch_sizes": batch_sizes,
            "rollout_steps": args.rollout_steps,
            "n_rollouts": args.n_rollouts,
            "results": [asdict(r) for r in results],
        }
        output_path = Path(args.output)
        with output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  Results appended to {output_path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI World Sim policy inference & rollout benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds to run each micro-benchmark after warmup.")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="Warmup seconds (excluded from timing).")
    parser.add_argument("--world-size", type=int, default=64,
                        help="World width and height in tiles.")
    parser.add_argument("--seed", type=int, default=42,
                        help="World generation seed.")
    parser.add_argument("--no-animals", action="store_true",
                        help="Disable scripted animals to isolate policy/env overhead.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device string (cpu / cuda / mps).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128, 512],
                        help="Batch sizes for the batched forward benchmark (B=1 is always run separately).")
    parser.add_argument("--rollout-steps", type=int, default=512,
                        help="Steps per rollout in the rollout throughput benchmark.")
    parser.add_argument("--n-rollouts", type=int, default=4,
                        help="Number of rollouts to run in the rollout benchmark.")
    parser.add_argument("--output", metavar="FILE",
                        help="Append a JSON-lines record to FILE for tracking over time.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
