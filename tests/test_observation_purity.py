"""Tests that observation building is pure — no side effects on memory."""

from __future__ import annotations

import numpy as np

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.rl.observations import build_observation, build_memory_features
from ai_world_sim.world.sim import WorldSim


def _make_sim():
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    cfg["animals"]["wolves_per_world"] = 0
    cfg["animals"]["rabbits_per_world"] = 0
    cfg["animals"]["deer_per_world"] = 0
    s = WorldSim(config=cfg, seed=42)
    s.generate()
    return s


def test_build_observation_is_pure():
    """Calling build_observation twice must not change memory state."""
    sim = _make_sim()
    agent = sim.spawn_agents(1)[0]

    # Capture memory state before first obs.
    food_count_before = len(agent.memory.known_food)
    water_count_before = len(agent.memory.known_water)

    obs1 = build_observation(sim, agent)
    food_count_mid = len(agent.memory.known_food)
    water_count_mid = len(agent.memory.known_water)

    obs2 = build_observation(sim, agent)
    food_count_after = len(agent.memory.known_food)
    water_count_after = len(agent.memory.known_water)

    # Memory must not change between calls.
    assert food_count_mid == food_count_before
    assert water_count_mid == water_count_before
    assert food_count_after == food_count_before
    assert water_count_after == water_count_before

    # Both observations must be identical (deterministic).
    np.testing.assert_array_equal(obs1["memory_features"], obs2["memory_features"])


def test_memory_updated_by_tick_not_observation():
    """Memory should only grow after tick(), not after build_observation()."""
    sim = _make_sim()
    agent = sim.spawn_agents(1)[0]

    # Clear memory (spawn already did initial perception — clear for clean slate).
    agent.memory.known_food.clear()
    agent.memory.known_water.clear()
    agent.memory.known_danger.clear()

    # build_observation must not update memory.
    _ = build_observation(sim, agent)
    assert len(agent.memory.known_food) == 0
    assert len(agent.memory.known_water) == 0

    # tick() must trigger a perception update.
    sim.tick()
    total_known = (
        len(agent.memory.known_food)
        + len(agent.memory.known_water)
        + len(agent.memory.known_danger)
    )
    # The agent should have perceived something in the 21×21 window.
    assert total_known > 0


def test_update_agent_memory_idempotent_within_tick():
    """Calling update_agent_memory twice at the same tick should not add duplicates."""
    sim = _make_sim()
    agent = sim.spawn_agents(1)[0]

    sim.update_agent_memory(agent)
    count_after_first = len(agent.memory.known_food) + len(agent.memory.known_water)

    sim.update_agent_memory(agent)
    count_after_second = len(agent.memory.known_food) + len(agent.memory.known_water)

    assert count_after_second == count_after_first
