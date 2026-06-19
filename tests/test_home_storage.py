"""Tests for the home storage (store_food / retrieve_food) system."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim


@pytest.fixture()
def small_config():
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    cfg["animals"]["wolves_per_world"] = 0
    cfg["animals"]["rabbits_per_world"] = 0
    cfg["animals"]["deer_per_world"] = 0
    return cfg


@pytest.fixture()
def sim(small_config):
    s = WorldSim(config=small_config, seed=42)
    s.generate()
    return s


def test_store_food_works_at_home(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.is_at_home

    agent.add_item("berries", 5)
    result = sim.store_food(agent)

    assert result is True
    assert agent.item_count("berries") == 0
    assert agent.stored_food.get("berries", 0) == 5


def test_store_food_fails_away_from_home(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.add_item("berries", 5)

    # Move agent away from home.
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = agent.position[0] + dr, agent.position[1] + dc
        if sim.is_passable(nr, nc):
            agent.position = (nr, nc)
            break

    assert not agent.is_at_home
    result = sim.store_food(agent)
    assert result is False
    assert agent.item_count("berries") == 5  # unchanged


def test_store_food_fails_without_food(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.is_at_home
    assert not agent.has_food()
    result = sim.store_food(agent)
    assert result is False


def test_retrieve_food_works_at_home(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.is_at_home

    # Pre-load stored food.
    agent.stored_food["berries"] = 10
    result = sim.retrieve_food(agent)

    assert result is True
    assert agent.item_count("berries") == 10
    assert not agent.has_stored_food()


def test_retrieve_food_fails_away_from_home(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.stored_food["berries"] = 5

    # Move away.
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = agent.position[0] + dr, agent.position[1] + dc
        if sim.is_passable(nr, nc):
            agent.position = (nr, nc)
            break

    assert not agent.is_at_home
    result = sim.retrieve_food(agent)
    assert result is False
    assert agent.stored_food.get("berries", 0) == 5


def test_retrieve_food_fails_without_stored_food(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.is_at_home
    assert not agent.has_stored_food()
    result = sim.retrieve_food(agent)
    assert result is False


def test_store_then_retrieve_roundtrip(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.add_item("berries", 4)
    agent.add_item("meat", 2)

    sim.store_food(agent)
    assert not agent.has_food()

    sim.retrieve_food(agent)
    assert agent.item_count("berries") == 4
    assert agent.item_count("meat") == 2


def test_action_mask_store_blocked_away_from_home(sim):
    from ai_world_sim.rl.observations import build_action_mask, STORE_FOOD

    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.add_item("berries", 3)

    # Move away from home.
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = agent.position[0] + dr, agent.position[1] + dc
        if sim.is_passable(nr, nc):
            agent.position = (nr, nc)
            break

    mask = build_action_mask(sim, agent)
    assert mask[STORE_FOOD] == 0.0


def test_action_mask_store_valid_at_home_with_food(sim):
    from ai_world_sim.rl.observations import build_action_mask, STORE_FOOD

    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.is_at_home
    agent.add_item("berries", 3)

    mask = build_action_mask(sim, agent)
    assert mask[STORE_FOOD] == 1.0


def test_action_mask_retrieve_valid_at_home_with_stored(sim):
    from ai_world_sim.rl.observations import build_action_mask, RETRIEVE_FOOD

    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.stored_food["berries"] = 5

    mask = build_action_mask(sim, agent)
    assert mask[RETRIEVE_FOOD] == 1.0
