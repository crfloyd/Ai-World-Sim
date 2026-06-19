"""Observation builder — converts raw world state into structured policy inputs.

Observation is a dict with four keys:

  "local_grid"     — (NUM_CHANNELS, WINDOW, WINDOW) float32 tensor
  "self_features"  — (SELF_DIM,) float32 vector
  "memory_features"— (MEMORY_DIM,) float32 vector
  "action_mask"    — (NUM_ACTIONS,) float32 binary mask

Spatial and scalar inputs are kept separate so the model can use the
appropriate encoder (CNN vs MLP) without flattening everything together.

Channel layout (NUM_CHANNELS = 15):
  Terrain   0-3  : grass, forest, mountain, water  (one-hot)
  Soil      4-6  : poor, normal, fertile            (one-hot)
  Resources 7-9  : berries, trees, stone            (normalised counts)
  Entities 10-12 : agent, prey, predator            (binary presence)
  Structure  13  : home                             (binary)
  Utility    14  : out_of_bounds                    (binary)

Self features (SELF_DIM = 14):
  0  hp              1  hunger       2  thirst       3  tired
  4  berries_inv     5  meat_inv     6  stored_food
  7  is_at_home      8  dist_home
  9-12 season one-hot (spring summer autumn winter)
  13 day_progress

Memory features (MEMORY_DIM = 12):
  See MemoryStore.summarize() for layout.

Actions (NUM_ACTIONS = 12):
  0 move_north   1 move_south   2 move_east    3 move_west
  4 forage       5 hunt         6 drink        7 eat
  8 rest         9 sleep       10 store_food  11 retrieve_food

TODO: Add "other agent" layer to CH_ENTITY_AGENT for multi-agent setup.
TODO: Add fog-of-war mask that blacks out unseen cells.
TODO: Add day/night cycle channel.
"""

from __future__ import annotations

import math

import numpy as np

from ai_world_sim.world.animals import Animal
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.memory import MEMORY_DIM
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.systems.seasons import Season
from ai_world_sim.world.terrain import (
    NUM_CHANNELS,
    SoilQuality,
    TerrainType,
    CH_TERRAIN_GRASS,
    CH_TERRAIN_FOREST,
    CH_TERRAIN_MOUNTAIN,
    CH_TERRAIN_WATER,
    CH_SOIL_POOR,
    CH_SOIL_NORMAL,
    CH_SOIL_FERTILE,
    CH_RESOURCE_BERRIES,
    CH_RESOURCE_TREES,
    CH_RESOURCE_STONE,
    CH_ENTITY_AGENT,
    CH_ENTITY_PREY,
    CH_ENTITY_PREDATOR,
    CH_STRUCTURE_HOME,
    CH_OUT_OF_BOUNDS,
)

NUM_SEASONS = len(Season)
SELF_DIM = 14
NUM_ACTIONS = 12

# Action index constants — single source of truth.
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
FORAGE = 4
HUNT = 5
DRINK = 6
EAT = 7
REST = 8
SLEEP = 9
STORE_FOOD = 10
RETRIEVE_FOOD = 11

_MOVE_DELTAS = {
    MOVE_NORTH: (-1, 0),
    MOVE_SOUTH: (1, 0),
    MOVE_EAST: (0, 1),
    MOVE_WEST: (0, -1),
}


def build_grid_tensor(
    world: WorldSim,
    agent: Agent,
    window: int,
) -> np.ndarray:
    """Return the (NUM_CHANNELS, WINDOW, WINDOW) float32 grid tensor.

    The agent is always centred in the window.  Out-of-bounds cells are
    encoded with CH_OUT_OF_BOUNDS=1 and all other channels=0.
    """
    half = window // 2
    r0, c0 = agent.position
    grid = np.zeros((NUM_CHANNELS, window, window), dtype=np.float32)

    cfg_res = world.config.get("resources", {})
    max_berries = float(cfg_res.get("max_berries", 5) or 1)
    max_trees = float(cfg_res.get("max_trees", 10) or 1)
    max_stone = float(cfg_res.get("max_stone", 8) or 1)

    # Build entity presence maps from animal positions.
    prey_positions: set[tuple[int, int]] = set()
    predator_positions: set[tuple[int, int]] = set()
    for animal in world.animals:
        if animal.alive:
            if animal.is_prey:
                prey_positions.add(animal.position)
            elif animal.is_predator:
                predator_positions.add(animal.position)

    for wr in range(window):
        for wc in range(window):
            nr = r0 + (wr - half)
            nc = c0 + (wc - half)

            if not world.in_bounds(nr, nc):
                grid[CH_OUT_OF_BOUNDS, wr, wc] = 1.0
                continue

            cell = world.grid[nr][nc]
            pos = (nr, nc)

            # Terrain (one-hot)
            ch_terrain = (
                CH_TERRAIN_GRASS if cell.terrain == TerrainType.GRASS
                else CH_TERRAIN_FOREST if cell.terrain == TerrainType.FOREST
                else CH_TERRAIN_MOUNTAIN if cell.terrain == TerrainType.MOUNTAIN
                else CH_TERRAIN_WATER
            )
            grid[ch_terrain, wr, wc] = 1.0

            # Soil (one-hot)
            ch_soil = (
                CH_SOIL_POOR if cell.soil == SoilQuality.POOR
                else CH_SOIL_NORMAL if cell.soil == SoilQuality.NORMAL
                else CH_SOIL_FERTILE
            )
            grid[ch_soil, wr, wc] = 1.0

            # Resources (normalised)
            grid[CH_RESOURCE_BERRIES, wr, wc] = cell.berries / max_berries
            grid[CH_RESOURCE_TREES, wr, wc] = cell.trees / max_trees
            grid[CH_RESOURCE_STONE, wr, wc] = cell.stone / max_stone

            # Entities
            if pos in prey_positions:
                grid[CH_ENTITY_PREY, wr, wc] = 1.0
            if pos in predator_positions:
                grid[CH_ENTITY_PREDATOR, wr, wc] = 1.0

            # Home structure
            if pos == agent.home_position:
                grid[CH_STRUCTURE_HOME, wr, wc] = 1.0

    return grid


def build_self_features(world: WorldSim, agent: Agent) -> np.ndarray:
    """Return the (SELF_DIM,) float32 self-feature vector."""
    agent_cfg = world.config.get("agents", {})
    max_hp = float(agent_cfg.get("max_hp", 100.0))
    max_hunger = float(agent_cfg.get("max_hunger", 100.0))
    max_thirst = float(agent_cfg.get("max_thirst", 100.0))
    max_tired = float(agent_cfg.get("max_tired", 100.0))

    season_one_hot = np.zeros(NUM_SEASONS, dtype=np.float32)
    season_one_hot[world.season] = 1.0

    year_length = world._season_sys.days_per_season * NUM_SEASONS
    day_progress = (world.day % max(year_length, 1)) / max(year_length, 1)

    stored_total = sum(agent.stored_food.values())

    # Distance to home, normalised by world diagonal.
    dr = agent.home_position[0] - agent.position[0]
    dc = agent.home_position[1] - agent.position[1]
    dist_home = math.sqrt(dr * dr + dc * dc) / max(world.world_diag, 1.0)

    return np.array(
        [
            agent.hp / max_hp,
            agent.hunger / max_hunger,
            agent.thirst / max_thirst,
            agent.tired / max_tired,
            math.log1p(agent.item_count("berries")) / math.log1p(20),
            math.log1p(agent.item_count("meat")) / math.log1p(20),
            math.log1p(stored_total) / math.log1p(50),
            1.0 if agent.is_at_home else 0.0,
            float(min(1.0, dist_home)),
            *season_one_hot,
            float(day_progress),
        ],
        dtype=np.float32,
    )


def build_memory_features(world: WorldSim, agent: Agent) -> np.ndarray:
    """Return the memory summary for *agent* (read-only; no side effects)."""
    return agent.memory.summarize(agent.position, world.world_diag)


def build_action_mask(world: WorldSim, agent: Agent) -> np.ndarray:
    """Return a (NUM_ACTIONS,) float32 binary validity mask.

    1.0 = action is valid, 0.0 = blocked.
    At least one action is always valid (rest is the fallback).
    """
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    r, c = agent.position

    # Movement: blocked by impassable terrain or world boundary.
    if not world.is_passable(r - 1, c):
        mask[MOVE_NORTH] = 0.0
    if not world.is_passable(r + 1, c):
        mask[MOVE_SOUTH] = 0.0
    if not world.is_passable(r, c + 1):
        mask[MOVE_EAST] = 0.0
    if not world.is_passable(r, c - 1):
        mask[MOVE_WEST] = 0.0

    # Forage: current cell must have resources.
    if not world.grid[r][c].has_forageable():
        mask[FORAGE] = 0.0

    # Hunt: must have adjacent living prey.
    if world.adjacent_prey(r, c) is None:
        mask[HUNT] = 0.0

    # Drink: must be adjacent to water.
    if not world.adjacent_water(r, c):
        mask[DRINK] = 0.0

    # Eat: must carry food.
    if not agent.has_food():
        mask[EAT] = 0.0

    # Rest / Sleep: always valid.

    # Store food: at home with food in inventory.
    if not (agent.is_at_home and agent.has_food()):
        mask[STORE_FOOD] = 0.0

    # Retrieve food: at home with stored food.
    if not (agent.is_at_home and agent.has_stored_food()):
        mask[RETRIEVE_FOOD] = 0.0

    # Safety fallback.
    if mask.sum() == 0.0:
        mask[REST] = 1.0

    return mask


def build_observation(
    world: WorldSim,
    agent: Agent,
    window: int = 21,
) -> dict[str, np.ndarray]:
    """Return the full observation dict for *agent*."""
    return {
        "local_grid": build_grid_tensor(world, agent, window),
        "self_features": build_self_features(world, agent),
        "memory_features": build_memory_features(world, agent),
        "action_mask": build_action_mask(world, agent),
    }
