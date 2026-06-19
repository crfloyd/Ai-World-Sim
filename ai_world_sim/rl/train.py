"""RLlib PPO training script.

Usage:
    python -m ai_world_sim.rl.train
    python -m ai_world_sim.rl.train --config configs/train.yaml

The ActionMaskModel wrapper reads the four-key observation dict produced by
WorldEnv and routes each component to the correct encoder in AgentBrain:
  "local_grid"      → CNN encoder
  "self_features"   → self MLP encoder
  "memory_features" → memory MLP encoder
  "action_mask"     → applied to logits as -1e9 for invalid actions

RLlib preprocessing is disabled (_disable_preprocessor_api=True) so the
dict observation passes through unchanged to the model forward().

TODO: Add curriculum: easy worlds first, increase scarcity over iterations.
TODO: Log metrics to W&B or TensorBoard.
TODO: Add validation callbacks that run frozen-policy episodes periodically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
import ray

from ai_world_sim.common.config import (
    DEFAULT_TRAIN_CONFIG_PATH,
    DEFAULT_WORLD_CONFIG_PATH,
    load_config,
)
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.model import AgentBrain
from ai_world_sim.rl.observations import NUM_ACTIONS


# ------------------------------------------------------------------ #
# RLlib model wrapper
# ------------------------------------------------------------------ #


class ActionMaskModel(TorchModelV2, nn.Module):
    """RLlib wrapper around AgentBrain.

    The observation dict passes through RLlib unchanged (preprocessor
    disabled) and is split here into grid / self / memory / mask.
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

        cc = model_config.get("custom_model_config", {})
        self.brain = AgentBrain(
            cnn_channels=cc.get("cnn_channels", [32, 64, 64]),
            use_global_avg_pool=cc.get("use_global_avg_pool", True),
            self_mlp_hidden=cc.get("self_mlp_hidden", [128, 64]),
            memory_mlp_hidden=cc.get("memory_mlp_hidden", [64, 32]),
            trunk_hidden=cc.get("trunk_hidden", [256, 128]),
        )
        self._value_out: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        local_grid = obs["local_grid"]
        self_feat = obs["self_features"]
        mem_feat = obs["memory_features"]
        action_mask = obs["action_mask"]

        logits, value = self.brain(local_grid, self_feat, mem_feat, action_mask)
        self._value_out = value
        return logits, state

    def value_function(self):
        return self._value_out


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #


def build_ppo_config(train_cfg: dict) -> PPOConfig:
    t = train_cfg.get("training", {})
    e = train_cfg.get("environment", {})
    m = train_cfg.get("model", {})

    env_config = {
        "world_config_path": str(DEFAULT_WORLD_CONFIG_PATH),
        "max_steps_per_episode": int(e.get("max_steps_per_episode", 1000)),
        "seed_range": [0, 899_999],
    }

    custom_model_config = {
        "cnn_channels": m.get("cnn_channels", [32, 64, 64]),
        "use_global_avg_pool": m.get("use_global_avg_pool", True),
        "self_mlp_hidden": m.get("self_mlp_hidden", [128, 64]),
        "memory_mlp_hidden": m.get("memory_mlp_hidden", [64, 32]),
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
                "_disable_preprocessor_api": True,
            },
        )
        .resources(num_gpus=int(os.environ.get("NUM_GPUS", 0)))
    )
    return config


def train(train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH) -> None:
    """Run the PPO training loop."""
    train_cfg = load_config(train_config_path)
    t = train_cfg.get("training", {})

    max_iter: int = int(t.get("max_iterations", 200))
    checkpoint_freq: int = int(t.get("checkpoint_freq", 50))
    checkpoint_dir: str = t.get("checkpoint_dir", "checkpoints/")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

    config = build_ppo_config(train_cfg)
    algo = config.build()

    print(f"Starting PPO training for {max_iter} iterations.")
    print(f"Checkpoints → {checkpoint_dir}")

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
    parser = argparse.ArgumentParser(description="Train AI World Sim with PPO.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAIN_CONFIG_PATH),
        help="Path to training YAML config.",
    )
    args = parser.parse_args()
    train(args.config)
