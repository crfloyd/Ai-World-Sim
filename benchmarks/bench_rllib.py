"""RLlib PPO throughput benchmark — worker and environment scaling sweep.

Measures how training throughput scales across combinations of rollout workers
and environments per worker, using the new RLlib API stack (RLModule / Learner).

Metrics per configuration
-------------------------
- env_steps/s       env_steps sampled per second (from RLlib's internal timer)
- learner_ms        learner update time per iteration (ms)
- reward_mean       mean episode return (NaN until first episode completes)
- cpu_pct           mean CPU utilisation during timed iterations (%)
- rss_mb            process RSS at end of measurement (MB)

Usage
-----
    python -m benchmarks.bench_rllib
    python -m benchmarks.bench_rllib --workers 1 2 4 --envs-per-worker 1 2 4
    python -m benchmarks.bench_rllib --iters 5 --train-batch-size 4000
    python -m benchmarks.bench_rllib --output results.jsonl

Flags
-----
    --workers           list of num_env_runners values    (default: 1 2 4)
    --envs-per-worker   list of num_envs_per_env_runner   (default: 1 2 4)
    --warmup-iters      iterations excluded from timing   (default: 1)
    --iters             measurement iterations            (default: 3)
    --train-batch-size  PPO train_batch_size              (default: 2000)
    --world-size        world width/height in tiles       (default: 64)
    --seed              world generation seed             (default: 42)
    --no-animals        disable scripted animals
    --output            append JSON-lines record to FILE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import ray
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_PER_SECOND,
    TIMERS,
)

from ai_world_sim.common.config import DEFAULT_WORLD_CONFIG_PATH, load_config
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.model import AgentBrain


# ---------------------------------------------------------------------------
# RLModule wrapper
# ---------------------------------------------------------------------------

class AgentBrainRLModule(TorchRLModule, ValueFunctionAPI):
    """TorchRLModule wrapping AgentBrain for the new RLlib API stack.

    Handles the four-key Dict observation space produced by WorldEnv:
      local_grid      → CNN encoder
      self_features   → self MLP encoder
      memory_features → memory MLP encoder
      action_mask     → applied as -1e9 logit mask before returning

    Implements ValueFunctionAPI so PPO can compute GAE without running the
    full network twice: _forward_train returns EMBEDDINGS (trunk output),
    and compute_values runs only the value head on those embeddings.
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        cc = self.model_config or {}
        self.brain = AgentBrain(
            cnn_channels=cc.get("cnn_channels", [32, 64, 64]),
            use_global_avg_pool=cc.get("use_global_avg_pool", True),
            self_mlp_hidden=cc.get("self_mlp_hidden", [128, 64]),
            memory_mlp_hidden=cc.get("memory_mlp_hidden", [64, 32]),
            trunk_hidden=cc.get("trunk_hidden", [256, 128]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, obs: dict) -> torch.Tensor:
        """Run through encoders + trunk; return trunk output (B, trunk_dim)."""
        cnn_embed = self.brain.cnn(obs["local_grid"])
        self_embed = F.relu(self.brain.self_encoder(obs["self_features"]))
        mem_embed = F.relu(self.brain.memory_encoder(obs["memory_features"]))
        combined = torch.cat([cnn_embed, self_embed, mem_embed], dim=-1)
        return self.brain.trunk(combined)

    def _logits_from_trunk(self, trunk_out: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.brain.policy_head(trunk_out)
        return logits + (1.0 - action_mask) * -1e9

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    @override(TorchRLModule)
    def _forward(self, batch: dict, **kwargs) -> dict:
        obs = batch[Columns.OBS]
        trunk_out = self._encode(obs)
        logits = self._logits_from_trunk(trunk_out, obs["action_mask"])
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch: dict, **kwargs) -> dict:
        obs = batch[Columns.OBS]
        trunk_out = self._encode(obs)
        logits = self._logits_from_trunk(trunk_out, obs["action_mask"])
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: trunk_out,
        }

    # ------------------------------------------------------------------
    # ValueFunctionAPI
    # ------------------------------------------------------------------

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict, embeddings: Any = None) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._encode(batch[Columns.OBS])
        return self.brain.value_head(embeddings).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-configuration result
# ---------------------------------------------------------------------------

@dataclass
class RllibResult:
    num_workers: int
    num_envs_per_worker: int
    steps_per_sec: float        # env_steps sampled / second
    learner_ms: float           # learner update time per iteration (ms)
    reward_mean: float          # episode_return_mean (NaN until first ep)
    cpu_pct: float              # mean CPU % during timed iters
    rss_mb: float               # RSS at end (MB)
    n_iters: int                # measurement iterations

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Single-configuration runner
# ---------------------------------------------------------------------------

def _build_env_config(world_cfg: dict, seed: int) -> dict:
    training_range = world_cfg.get("seeds", {}).get("training_range", [0, 899_999])
    return {
        "world_config_override": {
            "world": world_cfg.get("world", {}),
            "animals": world_cfg.get("animals", {}),
            "memory": world_cfg.get("memory", {}),
            "agents": world_cfg.get("agents", {}),
            "rewards": world_cfg.get("rewards", {}),
        },
        "max_steps_per_episode": 1000,
        "seed_range": training_range,
    }


def run_config(
    num_workers: int,
    num_envs: int,
    world_cfg: dict,
    seed: int,
    train_batch_size: int,
    warmup_iters: int,
    n_iters: int,
) -> RllibResult:
    """Train PPO for warmup + measurement iterations; return collected metrics."""
    env_config = _build_env_config(world_cfg, seed)

    config = (
        PPOConfig()
        .environment(WorldEnv, env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(module_class=AgentBrainRLModule)
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=min(512, train_batch_size),
            num_epochs=1,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
        )
        .resources(num_gpus=0)
    )

    algo = config.build()
    proc = psutil.Process(os.getpid())

    try:
        # Warmup — not counted
        for _ in range(warmup_iters):
            algo.train()

        # Measurement
        steps_per_sec_samples: list[float] = []
        learner_ms_samples: list[float] = []
        reward_samples: list[float] = []
        cpu_samples: list[float] = []

        psutil.cpu_percent(interval=None)  # prime the CPU counter

        for _ in range(n_iters):
            t0 = time.perf_counter()
            result = algo.train()
            wall_s = time.perf_counter() - t0

            # Throughput: prefer RLlib's own measurement; fall back to computed
            sps = result.get(NUM_ENV_STEPS_SAMPLED_PER_SECOND)
            if not sps or math.isnan(sps):
                steps_this_iter = result.get("num_env_steps_sampled_this_iter", 0) or 0
                sps = steps_this_iter / wall_s if wall_s > 0 else 0.0
            steps_per_sec_samples.append(sps)

            # Learner update time
            learner_s = result.get(TIMERS, {}).get(LEARNER_UPDATE_TIMER, 0.0) or 0.0
            learner_ms_samples.append(learner_s * 1000.0)

            # Episode reward (may be NaN / absent until an episode finishes)
            env_runner_results = result.get(ENV_RUNNER_RESULTS, {}) or {}
            rew = env_runner_results.get(EPISODE_RETURN_MEAN, float("nan"))
            if rew is not None and not math.isnan(rew):
                reward_samples.append(rew)

            cpu_samples.append(psutil.cpu_percent(interval=None))

        rss_mb = proc.memory_info().rss / 1e6
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

        return RllibResult(
            num_workers=num_workers,
            num_envs_per_worker=num_envs,
            steps_per_sec=avg(steps_per_sec_samples),
            learner_ms=avg(learner_ms_samples),
            reward_mean=avg(reward_samples) if reward_samples else float("nan"),
            cpu_pct=avg(cpu_samples),
            rss_mb=rss_mb,
            n_iters=n_iters,
        )
    finally:
        algo.stop()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_NAN_STR = "   ─   "


def _fmt_sps(v: float) -> str:
    return f"{v:>9,.0f}" if not math.isnan(v) else _NAN_STR


def _fmt_ms(v: float) -> str:
    return f"{v:>7.1f}ms" if not math.isnan(v) else _NAN_STR


def _fmt_rew(v: float) -> str:
    return f"{v:>7.3f}" if not math.isnan(v) else "    NaN"


def print_matrix(results: list[RllibResult], workers: list[int], envs: list[int]) -> None:
    """Print env_steps/s as a workers × envs/worker matrix."""
    lookup: dict[tuple[int, int], float] = {
        (r.num_workers, r.num_envs_per_worker): r.steps_per_sec for r in results
    }
    col_w = 12
    header = f"  {'workers/envs':>14}" + "".join(f"{'e='+str(e):>{col_w}}" for e in envs)
    print(header)
    print("  " + "-" * (14 + col_w * len(envs)))
    for w in workers:
        row = f"  {'w='+str(w):>14}"
        for e in envs:
            sps = lookup.get((w, e), float("nan"))
            row += f"{_fmt_sps(sps):>{col_w}}"
        print(row)


def print_detail_table(results: list[RllibResult]) -> None:
    hdr = f"  {'Config':>12}  {'steps/s':>10}  {'learn':>9}  {'reward':>8}  {'cpu%':>5}  {'rss':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        cfg = f"w={r.num_workers},e={r.num_envs_per_worker}"
        print(
            f"  {cfg:>12}  {_fmt_sps(r.steps_per_sec):>10}  "
            f"{_fmt_ms(r.learner_ms):>9}  {_fmt_rew(r.reward_mean):>8}  "
            f"{r.cpu_pct:>5.1f}  {r.rss_mb:>6.0f}M"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    world_cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    world_cfg["world"]["width"] = args.world_size
    world_cfg["world"]["height"] = args.world_size
    if args.no_animals:
        world_cfg["animals"]["wolves_per_world"] = 0
        world_cfg["animals"]["rabbits_per_world"] = 0
        world_cfg["animals"]["deer_per_world"] = 0

    workers: list[int] = args.workers
    envs_per_worker: list[int] = args.envs_per_worker

    n_configs = len(workers) * len(envs_per_worker)
    print(f"\nAI World Sim — RLlib PPO scaling benchmark")
    print(f"  Python {sys.version.split()[0]}  |  {platform.machine()}")
    print(f"  Ray {ray.__version__}  |  device: cpu")
    print(
        f"  world: {args.world_size}×{args.world_size}  seed: {args.seed}  "
        f"train_batch: {args.train_batch_size}  "
        f"warmup: {args.warmup_iters}i  iters: {args.iters}i  "
        f"animals: {'off' if args.no_animals else 'on'}"
    )
    print(f"  configs: {n_configs}  workers: {workers}  envs/worker: {envs_per_worker}")
    print()

    ray.init(ignore_reinit_error=True, logging_level="ERROR",
             log_to_driver=False, include_dashboard=False)

    results: list[RllibResult] = []
    for w in workers:
        for e in envs_per_worker:
            label = f"w={w}, e={e}"
            print(f"  [{label}] running ...", flush=True)
            t0 = time.perf_counter()
            try:
                r = run_config(
                    num_workers=w,
                    num_envs=e,
                    world_cfg=world_cfg,
                    seed=args.seed,
                    train_batch_size=args.train_batch_size,
                    warmup_iters=args.warmup_iters,
                    n_iters=args.iters,
                )
            except Exception as exc:
                print(f"  [{label}] ERROR: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            print(f"  [{label}] {r.steps_per_sec:,.0f} steps/s  "
                  f"learn={r.learner_ms:.0f}ms  cpu={r.cpu_pct:.0f}%  "
                  f"({elapsed:.0f}s wall)")
            results.append(r)

    ray.shutdown()

    if not results:
        print("\nNo results collected.")
        return

    print()
    print("  env_steps/s matrix (workers × envs/worker)")
    print_matrix(results, workers, envs_per_worker)
    print()
    print("  Detailed results")
    print_detail_table(results)
    print()

    if args.output:
        record = {
            "benchmark": "bench_rllib",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.machine(),
            "ray_version": ray.__version__,
            "world_size": args.world_size,
            "seed": args.seed,
            "train_batch_size": args.train_batch_size,
            "warmup_iters": args.warmup_iters,
            "n_iters": args.iters,
            "no_animals": args.no_animals,
            "workers": workers,
            "envs_per_worker": envs_per_worker,
            "results": [r.as_dict() for r in results],
        }
        output_path = Path(args.output)
        with output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  Results appended to {output_path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI World Sim RLlib PPO scaling benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4],
                        help="num_env_runners values to sweep.")
    parser.add_argument("--envs-per-worker", nargs="+", type=int, default=[1, 2, 4],
                        help="num_envs_per_env_runner values to sweep.")
    parser.add_argument("--warmup-iters", type=int, default=1,
                        help="Warmup iterations excluded from timing.")
    parser.add_argument("--iters", type=int, default=3,
                        help="Measurement iterations per configuration.")
    parser.add_argument("--train-batch-size", type=int, default=2000,
                        help="PPO train_batch_size (total steps per iteration).")
    parser.add_argument("--world-size", type=int, default=64,
                        help="World width and height in tiles.")
    parser.add_argument("--seed", type=int, default=42,
                        help="World generation seed.")
    parser.add_argument("--no-animals", action="store_true",
                        help="Disable scripted animals to isolate core overhead.")
    parser.add_argument("--output", metavar="FILE",
                        help="Append a JSON-lines record to FILE for tracking over time.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
