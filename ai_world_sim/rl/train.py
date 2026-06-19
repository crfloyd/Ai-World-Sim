"""RLlib PPO training script.

Usage:
    python -m ai_world_sim.rl.train
    python -m ai_world_sim.rl.train --config configs/train.yaml

The script:
  1. Reads training and world config.
  2. Registers WorldEnv with RLlib.
  3. Builds a PPOConfig with a custom action-masking model.
  4. Runs training for max_iterations, saving checkpoints.

Action masking is handled by the ActionMaskModel wrapper below, which
reads "action_mask" from the observation dict and subtracts a large value
from invalid action logits before the policy head.

TODO: Switch to RLlib's new API stack (Algorithm.from_config) once it
      stabilises for custom env + custom model combos.
TODO: Add curriculum: start with small worlds, grow as policy improves.
TODO: Add evaluation callbacks to track emergent behaviour metrics.
TODO: Log metrics to W&B or TensorBoard.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
import torch
import torch.nn as nn

from ai_world_sim.common.config import (
    DEFAULT_TRAIN_CONFIG_PATH,
    DEFAULT_WORLD_CONFIG_PATH,
    load_config,
    merge_configs,
)
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.model import AgentBrain
from ai_world_sim.rl.observations import NUM_ACTIONS, obs_dim


# ------------------------------------------------------------------ #
# RLlib model wrapper with action masking
# ------------------------------------------------------------------ #


class ActionMaskModel(TorchModelV2, nn.Module):
    """RLlib-compatible wrapper around AgentBrain with action-mask support.

    The observation dict is expected to have keys:
      "obs"          — flat float32 vector of shape (OBS_DIM,)
      "action_mask"  — float32 vector of shape (NUM_ACTIONS,)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_cfg: dict = model_config.get("custom_model_config", {})
        window: int = custom_cfg.get("window", 5)
        cnn_channels = [tuple(x) for x in custom_cfg.get("cnn_channels", [(16, 3), (32, 3)])]
        mlp_hidden: list[int] = custom_cfg.get("mlp_hidden", [128, 64])
        trunk_hidden: list[int] = custom_cfg.get("trunk_hidden", [256, 128])

        self.brain = AgentBrain(
            window=window,
            cnn_channels=cnn_channels,
            mlp_hidden=mlp_hidden,
            trunk_hidden=trunk_hidden,
        )
        self._value_out: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["obs"]
        action_mask = input_dict["obs"]["action_mask"]
        logits, value = self.brain(obs, action_mask)
        self._value_out = value
        return logits, state

    def value_function(self):
        return self._value_out


# ------------------------------------------------------------------ #
# Training entry point
# ------------------------------------------------------------------ #


def build_ppo_config(train_cfg: dict, world_cfg: dict) -> PPOConfig:
    """Construct an RLlib PPOConfig from our YAML config dicts."""
    t = train_cfg.get("training", {})
    e = train_cfg.get("environment", {})
    m = train_cfg.get("model", {})
    window = int(e.get("observation_window", 5))

    env_config = {
        "world_config_path": str(DEFAULT_WORLD_CONFIG_PATH),
        "observation_window": window,
        "max_steps_per_episode": int(e.get("max_steps_per_episode", 500)),
        "seed_range": e.get("seed_range", [0, 999_999]),
    }

    custom_model_config: dict[str, Any] = {
        "window": window,
        "cnn_channels": m.get("cnn_filters", [[16, 3], [32, 3]]),
        "mlp_hidden": m.get("mlp_hidden", [128, 64]),
        "trunk_hidden": m.get("trunk_hidden", [256, 128]),
    }

    config = (
        PPOConfig()
        .environment(WorldEnv, env_config=env_config)
        .framework("torch")
        .rollouts(
            num_rollout_workers=int(t.get("num_rollout_workers", 2)),
            num_envs_per_worker=int(t.get("num_envs_per_worker", 1)),
        )
        .training(
            lr=float(t.get("lr", 3e-4)),
            train_batch_size=int(t.get("train_batch_size", 4000)),
            sgd_minibatch_size=int(t.get("sgd_minibatch_size", 512)),
            num_sgd_iter=int(t.get("num_sgd_iter", 10)),
            gamma=float(t.get("gamma", 0.99)),
            lambda_=float(t.get("lambda_gae", 0.95)),
            clip_param=float(t.get("clip_param", 0.2)),
            entropy_coeff=float(t.get("entropy_coeff", 0.01)),
            vf_loss_coeff=float(t.get("vf_loss_coeff", 0.5)),
            model={
                "custom_model": "action_mask_model",
                "custom_model_config": custom_model_config,
            },
        )
        .resources(num_gpus=int(os.environ.get("NUM_GPUS", 0)))
    )
    return config


def train(train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH) -> None:
    """Run PPO training loop."""
    train_cfg = load_config(train_config_path)
    world_cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    t = train_cfg.get("training", {})

    max_iter: int = int(t.get("max_iterations", 200))
    checkpoint_freq: int = int(t.get("checkpoint_freq", 50))
    checkpoint_dir: str = t.get("checkpoint_dir", "checkpoints/")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    config = build_ppo_config(train_cfg, world_cfg)
    algo = config.build()

    print(f"Starting PPO training for {max_iter} iterations.")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    for i in range(1, max_iter + 1):
        result = algo.train()
        mean_reward = result.get("episode_reward_mean", float("nan"))
        ep_len = result.get("episode_len_mean", float("nan"))
        print(f"  iter {i:4d} | reward_mean={mean_reward:7.3f} | ep_len={ep_len:.1f}")

        if i % checkpoint_freq == 0:
            path = algo.save(checkpoint_dir)
            print(f"  Checkpoint saved: {path}")

    algo.stop()
    ray.shutdown()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI World Sim agent with PPO.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAIN_CONFIG_PATH),
        help="Path to training YAML config.",
    )
    args = parser.parse_args()
    train(args.config)
