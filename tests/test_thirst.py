"""Tests for the thirst / drink mechanic."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.terrain import Cell, TerrainType, SoilQuality


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


def test_thirst_increases_each_tick(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    t0 = agent.thirst
    sim.tick()
    assert agent.thirst > t0


def test_thirst_increases_faster_than_hunger(small_config):
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    sim.tick()
    assert agent.thirst > agent.hunger


def test_dehydration_damages_hp(small_config):
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.thirst = 100.0
    hp_before = agent.hp
    sim.tick()
    assert agent.hp < hp_before


def test_dehydration_kills_agent(small_config):
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.thirst = 100.0
    agent.hunger = 0.0  # only dehydration active

    for _ in range(200):
        sim.tick()
        if not agent.alive:
            break
    assert not agent.alive


def test_drink_reduces_thirst(small_config):
    """Place agent adjacent to a water cell and verify drink works."""
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.thirst = 80.0

    # Inject a water tile adjacent to the agent.
    r, c = agent.position
    nr = r + 1 if r + 1 < sim.height else r - 1
    sim.grid[nr][c] = Cell(
        terrain=TerrainType.WATER,
        soil=SoilQuality.NORMAL,
    )

    result = sim.drink(agent)
    assert result is True
    assert agent.thirst < 80.0


def test_drink_fails_without_adjacent_water(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.thirst = 80.0

    # Make all four neighbours non-water.
    r, c = agent.position
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if sim.in_bounds(nr, nc) and sim.grid[nr][nc].is_water():
            # Replace with grass.
            sim.grid[nr][nc] = Cell(
                terrain=TerrainType.GRASS,
                soil=SoilQuality.NORMAL,
            )

    result = sim.drink(agent)
    # Only fails if no water was adjacent (might still be adjacent if all 4 were
    # already non-water from generation).
    assert agent.thirst == 80.0 or result is False


def test_action_mask_blocks_drink_without_water(small_config):
    from ai_world_sim.rl.observations import build_action_mask, DRINK
    from ai_world_sim.world.terrain import Cell, TerrainType, SoilQuality

    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Replace all adjacent cells with grass.
    r, c = agent.position
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if sim.in_bounds(nr, nc) and sim.grid[nr][nc].is_water():
            sim.grid[nr][nc] = Cell(terrain=TerrainType.GRASS, soil=SoilQuality.NORMAL)

    mask = build_action_mask(sim, agent)
    assert mask[DRINK] == 0.0


def test_action_mask_allows_drink_with_adjacent_water(small_config):
    from ai_world_sim.rl.observations import build_action_mask, DRINK
    from ai_world_sim.world.terrain import Cell, TerrainType, SoilQuality

    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Force one adjacent cell to be water.
    r, c = agent.position
    nr = r + 1 if r + 1 < sim.height else r - 1
    sim.grid[nr][c] = Cell(terrain=TerrainType.WATER, soil=SoilQuality.NORMAL)

    mask = build_action_mask(sim, agent)
    assert mask[DRINK] == 1.0
