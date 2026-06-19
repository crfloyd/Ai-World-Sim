"""Tests for scripted animal entities and behaviors."""

from __future__ import annotations

import pytest

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.world.animals import Animal, AnimalSpecies
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim


@pytest.fixture()
def small_config():
    cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)
    cfg["world"]["width"] = 16
    cfg["world"]["height"] = 16
    cfg["animals"]["rabbits_per_world"] = 5
    cfg["animals"]["deer_per_world"] = 3
    cfg["animals"]["wolves_per_world"] = 2
    return cfg


@pytest.fixture()
def sim(small_config):
    s = WorldSim(config=small_config, seed=42)
    s.generate()
    return s


def test_animals_spawn(sim):
    assert len(sim.animals) > 0


def test_animals_at_passable_positions(sim):
    for animal in sim.animals:
        r, c = animal.position
        assert sim.is_passable(r, c), f"{animal.species} spawned at impassable ({r},{c})"


def test_all_species_present(sim, small_config):
    species_found = {a.species for a in sim.animals}
    assert AnimalSpecies.RABBIT in species_found
    assert AnimalSpecies.WOLF in species_found


def test_rabbit_is_prey():
    r = Animal(id=0, position=(0, 0), species=AnimalSpecies.RABBIT, hp=20.0)
    assert r.is_prey is True
    assert r.is_predator is False


def test_wolf_is_predator():
    w = Animal(id=0, position=(0, 0), species=AnimalSpecies.WOLF, hp=80.0)
    assert w.is_predator is True
    assert w.is_prey is False


def test_deer_meat_yield():
    d = Animal(id=0, position=(0, 0), species=AnimalSpecies.DEER, hp=40.0)
    assert d.meat_yield == 3


def test_wolf_attacks_adjacent_agent(small_config):
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Place wolf adjacent and away from home.
    agent.position = (5, 5)
    agent.home_position = (0, 0)
    wolf = Animal.create(0, (5, 6), AnimalSpecies.WOLF, small_config)
    sim.animals.append(wolf)

    hp_before = agent.hp
    sim._animal_sys.tick(sim, sim.animals, sim.agents, None)
    assert agent.hp < hp_before


def test_hunt_kills_adjacent_prey(small_config):
    small_config["animals"]["rabbits_per_world"] = 0
    small_config["animals"]["deer_per_world"] = 0
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    # Place prey adjacent.
    r, c = agent.position
    prey_pos = (r, c + 1) if c + 1 < sim.width and sim.is_passable(r, c + 1) else (r + 1, c)
    rabbit = Animal.create(0, prey_pos, AnimalSpecies.RABBIT, small_config)
    sim.animals.append(rabbit)

    result = sim.hunt(agent)
    assert result is True
    assert not rabbit.alive
    assert agent.item_count("meat") >= 1


def test_hunt_fails_without_adjacent_prey(small_config):
    small_config["animals"]["rabbits_per_world"] = 0
    small_config["animals"]["deer_per_world"] = 0
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]
    # No animals exist, hunt should fail.
    result = sim.hunt(agent)
    assert result is False
    assert agent.item_count("meat") == 0


def test_animals_move_each_tick(small_config):
    """Animals should move at least some of the time over many ticks."""
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)

    initial_positions = {a.id: a.position for a in sim.animals}
    for _ in range(20):
        sim.tick()

    final_positions = {a.id: a.position for a in sim.animals}
    moved = sum(1 for aid, pos in final_positions.items() if pos != initial_positions[aid])
    # At least one animal should have moved.
    assert moved > 0


def test_action_mask_hunt_blocked_without_prey(small_config):
    from ai_world_sim.rl.observations import build_action_mask, HUNT

    small_config["animals"]["rabbits_per_world"] = 0
    small_config["animals"]["deer_per_world"] = 0
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    mask = build_action_mask(sim, agent)
    assert mask[HUNT] == 0.0


def test_action_mask_hunt_valid_with_adjacent_prey(small_config):
    from ai_world_sim.rl.observations import build_action_mask, HUNT

    small_config["animals"]["rabbits_per_world"] = 0
    small_config["animals"]["deer_per_world"] = 0
    small_config["animals"]["wolves_per_world"] = 0
    sim = WorldSim(config=small_config, seed=42)
    sim.generate()
    agents = sim.spawn_agents(1)
    agent = agents[0]

    r, c = agent.position
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if sim.is_passable(nr, nc):
            rabbit = Animal.create(0, (nr, nc), AnimalSpecies.RABBIT, small_config)
            sim.animals.append(rabbit)
            mask = build_action_mask(sim, agent)
            assert mask[HUNT] == 1.0
            return
    pytest.skip("No passable adjacent cell for prey.")
