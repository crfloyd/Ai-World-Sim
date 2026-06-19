"""Tests for Agent entity and WorldSim agent interactions."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.memory import MemoryStore
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.terrain import TerrainType


@pytest.fixture()
def small_config():
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    cfg["animals"]["rabbits_per_world"] = 2
    cfg["animals"]["deer_per_world"] = 1
    cfg["animals"]["wolves_per_world"] = 0
    return cfg


@pytest.fixture()
def sim(small_config):
    s = WorldSim(config=small_config, seed=123)
    s.generate()
    return s


def test_agent_defaults():
    a = Agent(id=0, position=(5, 5), home_position=(5, 5))
    assert a.hp == 100.0
    assert a.hunger == 0.0
    assert a.thirst == 0.0
    assert a.tired == 0.0
    assert a.alive is True
    assert a.sleeping is False
    assert a.inventory == {}
    assert a.stored_food == {}
    assert isinstance(a.memory, MemoryStore)


def test_agent_is_at_home():
    a = Agent(id=0, position=(3, 4), home_position=(3, 4))
    assert a.is_at_home is True
    a.position = (3, 5)
    assert a.is_at_home is False


def test_agent_inventory_add_remove():
    a = Agent(id=1, position=(0, 0), home_position=(0, 0))
    a.add_item("berries", 3)
    assert a.item_count("berries") == 3
    assert a.total_carried() == 3
    assert a.has_food() is True

    removed = a.remove_item("berries", 2)
    assert removed is True
    assert a.item_count("berries") == 1


def test_agent_inventory_remove_insufficient():
    a = Agent(id=2, position=(0, 0), home_position=(0, 0))
    a.add_item("berries", 1)
    assert a.remove_item("berries", 5) is False
    assert a.item_count("berries") == 1


def test_agent_home_storage():
    a = Agent(id=3, position=(0, 0), home_position=(0, 0))
    a.add_item("berries", 5)
    a.add_item("meat", 2)
    stored = a.store_all_food()
    assert stored == 7
    assert a.item_count("berries") == 0
    assert a.item_count("meat") == 0
    assert a.has_stored_food() is True

    retrieved = a.retrieve_all_food()
    assert retrieved == 7
    assert a.item_count("berries") == 5
    assert not a.has_stored_food()


def test_spawn_agents(sim):
    agents = sim.spawn_agents(2)
    assert len(agents) == 2
    assert len(sim.agents) == 2
    for agent in agents:
        r, c = agent.position
        assert sim.is_passable(r, c)
        assert agent.home_position == agent.position


def test_agent_move_valid(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    original_pos = agent.position
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr = original_pos[0] + dr
        nc = original_pos[1] + dc
        if sim.is_passable(nr, nc):
            success = sim.move_agent(agent, dr, dc)
            assert success is True
            assert agent.position == (nr, nc)
            return
    pytest.skip("No passable neighbour for spawn position.")


def test_agent_cannot_move_into_water(sim):
    for r in range(1, sim.height - 1):
        for c in range(1, sim.width - 1):
            if sim.grid[r][c].is_passable() and sim.grid[r - 1][c].terrain == TerrainType.WATER:
                agent = Agent(id=99, position=(r, c), home_position=(r, c))
                sim.agents.append(agent)
                assert sim.move_agent(agent, -1, 0) is False
                assert agent.position == (r, c)
                return
    pytest.skip("No passable-adjacent-to-water cell in this seed.")


def test_agent_hunger_increases_each_tick(small_config):
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    initial = agent.hunger
    sim.tick()
    assert agent.hunger > initial


def test_agent_thirst_increases_each_tick(small_config):
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    initial = agent.thirst
    sim.tick()
    assert agent.thirst > initial


def test_agent_tired_increases_each_tick(small_config):
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    initial = agent.tired
    sim.tick()
    assert agent.tired > initial


def test_agent_death_on_starvation(small_config):
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=7)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.hunger = 100.0

    for _ in range(300):
        sim.tick()
        if not agent.alive:
            break
    assert not agent.alive


def test_forage_adds_to_inventory(sim):
    """After the refactor, forage adds to inventory rather than eating directly."""
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Find a cell with berries and place agent there.
    for r in range(sim.height):
        for c in range(sim.width):
            if sim.grid[r][c].berries > 0:
                agent.position = (r, c)
                before = agent.item_count("berries")
                sim.forage(agent)
                assert agent.item_count("berries") > before
                return
    pytest.skip("No berry cell found.")


def test_eat_reduces_hunger(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.add_item("berries", 3)
    agent.hunger = 80.0
    sim.eat(agent)
    assert agent.hunger < 80.0
