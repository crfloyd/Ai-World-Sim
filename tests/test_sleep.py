"""Tests for the tired stat, rest action, and sleep action."""

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


def test_tired_increases_each_tick(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    t0 = agent.tired
    sim.tick()
    assert agent.tired > t0


def test_movement_increases_tired_extra(small_config):
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()

    a_moved = sim.spawn_agents(1)[0]
    a_rested = Agent(id=99, position=a_moved.position, home_position=a_moved.position)
    sim.agents.append(a_rested)

    # Both start at tired=0.
    # Move one agent and tick once; the other doesn't act.
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = a_moved.position[0] + dr, a_moved.position[1] + dc
        if sim.is_passable(nr, nc):
            sim.move_agent(a_moved, dr, dc)
            break

    assert a_moved.tired > a_rested.tired


def test_rest_reduces_tired(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    agent.tired = 50.0
    sim.rest(agent)
    assert agent.tired < 50.0


def test_sleep_reduces_tired_more_than_rest(small_config):
    sim_a = WorldSim(config=small_config, seed=42)
    sim_a.generate()
    agent_rest = sim_a.spawn_agents(1)[0]
    agent_rest.tired = 80.0
    sim_a.rest(agent_rest)

    sim_b = WorldSim(config=small_config, seed=42)
    sim_b.generate()
    agent_sleep = sim_b.spawn_agents(1)[0]
    agent_sleep.tired = 80.0
    sim_b.sleep(agent_sleep)

    assert agent_sleep.tired < agent_rest.tired


def test_sleep_sets_sleeping_flag(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    assert agent.sleeping is False
    sim.sleep(agent)
    assert agent.sleeping is True


def test_non_sleep_action_clears_sleeping(sim):
    agents = sim.spawn_agents(1)
    agent = agents[0]
    sim.sleep(agent)
    assert agent.sleeping is True
    sim.rest(agent)
    assert agent.sleeping is False


def test_sleep_at_home_recovers_more(small_config):
    """Sleep at home_position should give more recovery than outside."""
    sim_home = WorldSim(config=small_config, seed=42)
    sim_home.generate()
    agent_home = sim_home.spawn_agents(1)[0]
    agent_home.tired = 80.0
    # agent_home is already at home_position (spawn = home)
    assert agent_home.is_at_home
    sim_home.sleep(agent_home)
    tired_after_home = agent_home.tired

    sim_away = WorldSim(config=small_config, seed=42)
    sim_away.generate()
    agent_away = sim_away.spawn_agents(1)[0]
    agent_away.tired = 80.0
    # Move agent away from home.
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = agent_away.position[0] + dr, agent_away.position[1] + dc
        if sim_away.is_passable(nr, nc):
            agent_away.position = (nr, nc)
            break
    assert not agent_away.is_at_home
    sim_away.sleep(agent_away)
    tired_after_away = agent_away.tired

    assert tired_after_home < tired_after_away


def test_wolf_deals_bonus_damage_to_sleeping_agent(small_config):
    """Wolf should deal extra damage when agent is sleeping outside home."""
    from ai_world_sim.world.animals import Animal, AnimalSpecies

    small_config["animals"]["wolves_per_world"] = 0  # we'll add one manually
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Move agent away from home.
    agent.position = (5, 5)
    agent.home_position = (0, 0)

    # Place wolf adjacent to agent.
    wolf_pos = (5, 6)
    wolf = Animal.create(0, wolf_pos, AnimalSpecies.WOLF, small_config)
    sim.animals.append(wolf)

    # Test 1: awake agent.
    agent.sleeping = False
    agent.hp = 100.0
    sim._animal_sys.tick(sim, sim.animals, sim.agents, None)
    damage_awake = 100.0 - agent.hp

    # Test 2: sleeping agent (same wolf, same position).
    agent.sleeping = True
    agent.hp = 100.0
    sim._animal_sys.tick(sim, sim.animals, sim.agents, None)
    damage_sleeping = 100.0 - agent.hp

    assert damage_sleeping > damage_awake


def test_wolf_does_not_attack_agent_sleeping_at_home(small_config):
    """Wolves respect home safety."""
    from ai_world_sim.world.animals import Animal, AnimalSpecies

    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Agent is at home and sleeping.
    agent.position = agent.home_position
    agent.sleeping = True
    agent.hp = 100.0

    # Place wolf adjacent.
    r, c = agent.position
    wolf_pos = (r, c + 1) if c + 1 < sim.width else (r, c - 1)
    wolf = Animal.create(0, wolf_pos, AnimalSpecies.WOLF, small_config)
    sim.animals.append(wolf)

    sim._animal_sys.tick(sim, sim.animals, sim.agents, None)
    assert agent.hp == 100.0  # no damage at home
