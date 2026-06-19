"""Tests for world generation and core sim mechanics."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.generator import generate_world
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.terrain import TerrainType


@pytest.fixture()
def world_config():
    return load_config(DEFAULT_WORLD_CONFIG_PATH)


@pytest.fixture()
def small_config(world_config):
    """Override dimensions to something tiny for fast tests."""
    cfg = dict(world_config)
    cfg["world"] = dict(world_config.get("world", {}))
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    return cfg


def test_deterministic_generation(small_config):
    """Same seed must produce identical grids."""
    g1 = generate_world(16, 16, seed=7, config=small_config)
    g2 = generate_world(16, 16, seed=7, config=small_config)

    for r in range(16):
        for c in range(16):
            assert g1[r][c].terrain == g2[r][c].terrain
            assert g1[r][c].soil == g2[r][c].soil
            assert g1[r][c].trees == g2[r][c].trees
            assert g1[r][c].berries == g2[r][c].berries


def test_different_seeds_differ(small_config):
    """Different seeds should (almost always) produce different grids."""
    g1 = generate_world(16, 16, seed=1, config=small_config)
    g2 = generate_world(16, 16, seed=9999, config=small_config)
    terrains1 = [g1[r][c].terrain for r in range(16) for c in range(16)]
    terrains2 = [g2[r][c].terrain for r in range(16) for c in range(16)]
    assert terrains1 != terrains2


def test_grid_dimensions(small_config):
    g = generate_world(20, 10, seed=42, config=small_config)
    assert len(g) == 10
    assert all(len(row) == 20 for row in g)


def test_terrain_types_valid(small_config):
    g = generate_world(16, 16, seed=42, config=small_config)
    valid = set(TerrainType)
    for row in g:
        for cell in row:
            assert cell.terrain in valid


def test_worldsim_generate(small_config):
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    assert sim.grid is not None
    assert len(sim.grid) == 16
    assert sim.day == 0
    assert sim.tick_count == 0


def test_worldsim_tick_advances_time(small_config):
    sim = WorldSim(config=small_config, seed=1)
    sim.generate()
    sim.spawn_agents(1)
    sim.tick()
    assert sim.tick_count == 1


def test_worldsim_advance_day(small_config):
    ticks_per_day = small_config.get("world", {}).get("ticks_per_day", 24)
    sim = WorldSim(config=small_config, seed=1)
    sim.generate()
    sim.spawn_agents(1)
    sim.advance_day()
    assert sim.day == 1
    assert sim.tick_count == ticks_per_day
