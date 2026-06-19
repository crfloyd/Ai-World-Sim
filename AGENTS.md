# AGENTS.md — AI World Sim

Knowledge base for agents and contributors working in this repo.

---

## Project Overview

AI World Sim is a PPO training project where a single shared neural policy (`AgentBrain`) learns to survive in a procedurally generated grid world. The spec lives in `docs/v0-survival-spec.md`. V0 is survival only — no farming, crafting, factions, or social systems.

**Core stack**: Python + PyTorch + Gymnasium + Ray RLlib 2.55.1

---

## Repository Layout

```
ai_world_sim/
  common/config.py          load_config(), DEFAULT_WORLD_CONFIG_PATH
  world/sim.py              WorldSim — tick(), spawn_agents(), update_agent_memory()
  world/entities.py         Agent dataclass
  world/memory.py           MemoryStore, MEMORY_DIM = 12
  world/terrain.py          terrain/soil channel constants
  rl/env.py                 WorldEnv (Gymnasium env)
  rl/model.py               AgentBrain (CNN + dual-MLP actor-critic)
  rl/observations.py        build_observation(), NUM_ACTIONS=12, NUM_CHANNELS=15, SELF_DIM=14
  rl/rewards.py             survival_reward(agent, died, stored_food, config)
  rl/train.py               RLlib training entry point (old API stack — needs update)
benchmarks/
  _utils.py                 BenchResult, _timed_loop, print_header/result
  bench_sim.py              WorldSim.tick / obs encoding / Gym env step throughput
  bench_policy.py           AgentBrain inference / rollout throughput
  bench_rllib.py            RLlib PPO scaling sweep (workers × envs/worker)
configs/
  world.yaml                all world + reward + seed config
  train.yaml                PPO hyperparams
tests/                      pytest suite (115 tests)
docs/v0-survival-spec.md    authoritative V0 spec
```

---

## Key Design Rules

- **Observation building is side-effect free.** `build_observation()` never mutates world or agent state. Memory is updated only in `WorldSim.tick()` (step 6) and `spawn_agents()`.
- **Reward weights live in config.** `configs/world.yaml` → `rewards:` section. `survival_reward()` reads from `config["rewards"]` with constants as fallback.
- **Bounded food retrieval.** `retrieve_food()` calls `agent.retrieve_up_to(N)` where N = `config["agents"]["retrieve_food_amount"]` (default 1). One unit per action.
- **All randomness must be seeded.** Deterministic replay is a requirement.
- **Seed pools**: training [0, 899999], validation 900001–900010, story 999001–999005. Ranges are in `world.yaml → seeds:`.
- **No V1+ features.** Farming, crafting, trade, factions, crime, social systems are all out of scope.
- **One action is always valid.** REST (action 8) is the unconditional fallback; action masking must never block all actions.

---

## AgentBrain Architecture

```
local_grid (15, 21, 21)  → Conv×3 + GlobalAvgPool → (B, 64)
self_features (14,)      → MLP [128, 64]           → (B, 64)
memory_features (12,)    → MLP [64, 32]            → (B, 32)
concat (B, 160) → shared trunk MLP [256, 128]
         → policy_head Linear(128, 12) + action mask (-1e9 on blocked)
         → value_head  Linear(128, 1)
```

`AgentBrain.forward()` returns `(logits, value)`. Action masking is applied as `logits += (1 - mask) * -1e9`.

`act(deterministic=True)` uses `torch.argmax` (not `dist.mode`).

---

## RLlib 2.55.1 — New API Stack

**TL;DR**: The new API stack is the default in 2.55.1. Use `TorchRLModule` + `ValueFunctionAPI`. Do not use `TorchModelV2`, `custom_model`, or `.rollouts()`.

### What changed

| Old (deprecated) | New (2.55.1) |
|---|---|
| `.rollouts(num_rollout_workers=N)` | `.env_runners(num_env_runners=N)` |
| `.rollouts(num_envs_per_worker=M)` | `.env_runners(num_envs_per_env_runner=M)` |
| `TorchModelV2` + `ModelCatalog` | `TorchRLModule` + `RLModuleSpec` |
| `custom_model` in `.training(model=...)` | `.rl_module(rl_module_spec=RLModuleSpec(...))` |
| `sgd_minibatch_size` | `minibatch_size` |
| `num_sgd_iter` | `num_epochs` |

`.rollouts()` raises `ValueError` in 2.55.1 — it cannot be used at all.

### Dict obs space

RLlib 2.55.1 has no default encoder for `gymnasium.spaces.Dict`. You must implement `TorchRLModule` and handle obs extraction manually.

### AgentBrainRLModule pattern

`benchmarks/bench_rllib.py` has `AgentBrainRLModule(TorchRLModule, ValueFunctionAPI)`:
- `setup()` — creates `AgentBrain` from `self.model_config`
- `_encode(obs)` — runs CNN + MLP encoders + shared trunk → trunk_out
- `_forward(batch)` — inference/exploration: `{ACTION_DIST_INPUTS: logits}`
- `_forward_train(batch)` — training: `{ACTION_DIST_INPUTS: logits, EMBEDDINGS: trunk_out}`
- `compute_values(batch, embeddings)` — runs value head on trunk_out

### Metric keys (result dict after `algo.train()`)

```python
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED_PER_SECOND,  # top-level
    ENV_RUNNER_RESULTS,                # = 'env_runners'
    EPISODE_RETURN_MEAN,               # inside env_runners
    TIMERS,                            # = 'timers'
    LEARNER_UPDATE_TIMER,              # inside timers (seconds)
)

sps   = result[NUM_ENV_STEPS_SAMPLED_PER_SECOND]
rew   = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
ltime = result[TIMERS][LEARNER_UPDATE_TIMER]  # seconds
```

### Minimal working PPOConfig

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

config = (
    PPOConfig()
    .environment(WorldEnv, env_config=env_config)
    .framework("torch")
    .env_runners(num_env_runners=2, num_envs_per_env_runner=2)
    .rl_module(rl_module_spec=RLModuleSpec(module_class=AgentBrainRLModule))
    .training(
        train_batch_size=4000,
        minibatch_size=512,
        num_epochs=10,
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
```

### train.py needs updating

`ai_world_sim/rl/train.py` still uses the old API (`.rollouts()`, `TorchModelV2`, `ModelCatalog`). It will fail in 2.55.1. The benchmark uses the new API. When updating `train.py`, follow `AgentBrainRLModule` as the reference pattern.

---

## Throughput Numbers (reference machine, CPU-only)

From bench_sim.py:
- `WorldSim.tick()`: ~47,000/s
- `build_observation()`: ~375/s
- `WorldEnv.step()`: ~375/s

From bench_policy.py:
- `AgentBrain.forward() [B=1]`: ~20,000/s (~50µs)
- `WorldEnv.step() [rand-valid]`: ~375/s
- `WorldEnv.step() [policy]`: ~360/s
- Policy rollout (512×4): ~350 samples/s

**Env stepping is the bottleneck**, not model inference. `build_observation()` dominates `WorldEnv.step()` cost. This means the parallelism from `num_env_runners` / `num_envs_per_env_runner` is the primary lever.

---

## Common Commands

```bash
# Run tests
pytest tests/ -q

# Run sim benchmarks
python -m benchmarks.bench_sim --world-size 32 --no-animals

# Run policy benchmarks
python -m benchmarks.bench_policy --world-size 32

# Run RLlib scaling benchmark (quick)
python -m benchmarks.bench_rllib --workers 1 2 --envs-per-worker 1 2 --iters 2

# Full RLlib scaling sweep
python -m benchmarks.bench_rllib --workers 1 2 4 8 --envs-per-worker 1 2 4 8 --output results.jsonl
```

---

## Branch Conventions

Development branches: `claude/<topic>`. All work is committed and pushed before a PR is opened. Do not create PRs unless explicitly requested.
