"""Tests for Agent entity and WorldSim agent interactions."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.terrain import TerrainType


@pytest.fixture()
def small_config():
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    return cfg


@pytest.fixture()
def sim(small_config):
    s = WorldSim(config=small_config, seed=123)
    s.generate()
    return s


def test_agent_defaults():
    a = Agent(id=0, position=(5, 5))
    assert a.hp == 100.0
    assert a.hunger == 0.0
    assert a.fatigue == 0.0
    assert a.alive is True
    assert a.inventory == {}


def test_agent_inventory_add_remove():
    a = Agent(id=1, position=(0, 0))
    a.add_item("berries", 3)
    assert a.item_count("berries") == 3
    assert a.total_carried() == 3

    removed = a.remove_item("berries", 2)
    assert removed is True
    assert a.item_count("berries") == 1


def test_agent_inventory_remove_insufficient():
    a = Agent(id=2, position=(0, 0))
    a.add_item("berries", 1)
    removed = a.remove_item("berries", 5)
    assert removed is False
    assert a.item_count("berries") == 1


def test_agent_item_count_zero_for_missing():
    a = Agent(id=3, position=(0, 0))
    assert a.item_count("gold") == 0


def test_spawn_agents(sim):
    agents = sim.spawn_agents(2)
    assert len(agents) == 2
    assert len(sim.agents) == 2
    for agent in agents:
        r, c = agent.position
        assert sim.is_passable(r, c)


def test_agent_move_valid(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Try moving in each direction until we find a valid one.
    original_pos = agent.position
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr = original_pos[0] + dr
        nc = original_pos[1] + dc
        if sim.is_passable(nr, nc):
            success = sim.move_agent(agent, dr, dc)
            assert success is True
            assert agent.position == (nr, nc)
            return
    pytest.skip("No passable neighbor found for agent spawn position.")


def test_agent_cannot_move_into_water(sim):
    """If we manufacture a scenario where the neighbor is water, movement fails."""
    # Find a passable cell adjacent to water manually.
    for r in range(1, sim.height - 1):
        for c in range(1, sim.width - 1):
            if sim.grid[r][c].is_passable() and sim.grid[r - 1][c].terrain == TerrainType.WATER:
                agent = Agent(id=99, position=(r, c))
                sim.agents.append(agent)
                result = sim.move_agent(agent, -1, 0)
                assert result is False
                assert agent.position == (r, c)
                return
    pytest.skip("No passable-cell-adjacent-to-water found in this seed.")


def test_agent_hunger_increases_each_tick(small_config):
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    initial_hunger = agent.hunger
    sim.tick()
    assert agent.hunger > initial_hunger


def test_agent_death_on_starvation(small_config):
    """Force an agent to 100 hunger and confirm HP drains and death occurs."""
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.hunger = 100.0  # already starving

    # Starvation should drain HP; enough ticks should kill the agent.
    for _ in range(200):
        sim.tick()
        if not agent.alive:
            break
    assert not agent.alive
