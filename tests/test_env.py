"""Tests for the Gymnasium RL environment."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.observations import NUM_ACTIONS, obs_dim


@pytest.fixture()
def env():
    e = WorldEnv(
        env_config={
            "observation_window": 5,
            "max_steps_per_episode": 50,
            "seed_range": [42, 42],
        }
    )
    return e


def test_observation_space_shape(env):
    obs, _ = env.reset(seed=42)
    assert "obs" in obs
    assert "action_mask" in obs
    expected_dim = obs_dim(5)
    assert obs["obs"].shape == (expected_dim,)
    assert obs["action_mask"].shape == (NUM_ACTIONS,)


def test_observation_dtype(env):
    obs, _ = env.reset(seed=42)
    assert obs["obs"].dtype == np.float32
    assert obs["action_mask"].dtype == np.float32


def test_action_mask_binary(env):
    obs, _ = env.reset(seed=42)
    mask = obs["action_mask"]
    assert set(mask).issubset({0.0, 1.0})


def test_action_mask_has_valid_action(env):
    """At least one action must always be valid."""
    obs, _ = env.reset(seed=42)
    assert obs["action_mask"].sum() >= 1


def test_step_returns_correct_types(env):
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(5)  # REST
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert int(env.action_space.n) == NUM_ACTIONS


def test_episode_terminates_on_truncation(env):
    env.reset(seed=1)
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps <= 50  # max_steps_per_episode in fixture


def test_reset_regenerates_world():
    """Two resets with the same seed should produce identical observations."""
    env = WorldEnv(env_config={"observation_window": 5, "max_steps_per_episode": 50})
    obs1, _ = env.reset(seed=77)
    obs2, _ = env.reset(seed=77)
    np.testing.assert_array_equal(obs1["obs"], obs2["obs"])
    np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])


def test_reset_different_seeds_differ():
    env = WorldEnv(env_config={"observation_window": 5, "max_steps_per_episode": 50})
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=9999)
    # Observations should differ between seeds (world is different).
    assert not np.array_equal(obs1["obs"], obs2["obs"])


def test_info_contains_expected_keys(env):
    env.reset(seed=42)
    _, _, _, _, info = env.step(5)
    assert "day" in info
    assert "season" in info
    assert "hp" in info
    assert "hunger" in info
    assert "fatigue" in info
