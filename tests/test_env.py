"""Tests for the Gymnasium RL environment."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.observations import NUM_ACTIONS, NUM_CHANNELS, SELF_DIM
from ai_world_sim.world.memory import MEMORY_DIM


@pytest.fixture()
def env():
    return WorldEnv(
        env_config={
            "max_steps_per_episode": 50,
            "seed_range": [42, 42],
        }
    )


def test_observation_space_keys(env):
    obs, _ = env.reset(seed=42)
    assert set(obs.keys()) == {"local_grid", "self_features", "memory_features", "action_mask"}


def test_observation_shapes(env):
    obs, _ = env.reset(seed=42)
    window = env.window  # 21 by default
    assert obs["local_grid"].shape == (NUM_CHANNELS, window, window)
    assert obs["self_features"].shape == (SELF_DIM,)
    assert obs["memory_features"].shape == (MEMORY_DIM,)
    assert obs["action_mask"].shape == (NUM_ACTIONS,)


def test_observation_dtypes(env):
    obs, _ = env.reset(seed=42)
    for key, arr in obs.items():
        assert arr.dtype == np.float32, f"{key} should be float32"


def test_grid_values_in_range(env):
    obs, _ = env.reset(seed=42)
    assert obs["local_grid"].min() >= 0.0
    assert obs["local_grid"].max() <= 1.0


def test_self_features_length(env):
    obs, _ = env.reset(seed=42)
    assert len(obs["self_features"]) == SELF_DIM


def test_action_mask_binary(env):
    obs, _ = env.reset(seed=42)
    mask = obs["action_mask"]
    assert set(mask).issubset({0.0, 1.0})


def test_action_mask_has_valid_action(env):
    obs, _ = env.reset(seed=42)
    assert obs["action_mask"].sum() >= 1


def test_action_space_size(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert int(env.action_space.n) == NUM_ACTIONS  # 12


def test_step_returns_correct_types(env):
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(8)  # REST
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_info_contains_expected_keys(env):
    env.reset(seed=42)
    _, _, _, _, info = env.step(8)
    for key in ("day", "season", "hp", "hunger", "thirst", "tired", "steps"):
        assert key in info


def test_episode_truncation(env):
    env.reset(seed=1)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps <= 50


def test_reset_same_seed_deterministic():
    e = WorldEnv(env_config={"max_steps_per_episode": 50})
    obs1, _ = e.reset(seed=77)
    obs2, _ = e.reset(seed=77)
    np.testing.assert_array_equal(obs1["local_grid"], obs2["local_grid"])
    np.testing.assert_array_equal(obs1["self_features"], obs2["self_features"])
    np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])


def test_reset_different_seeds_differ():
    e = WorldEnv(env_config={"max_steps_per_episode": 50})
    obs1, _ = e.reset(seed=1)
    obs2, _ = e.reset(seed=9999)
    assert not np.array_equal(obs1["local_grid"], obs2["local_grid"])
