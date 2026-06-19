"""Observation builder — converts raw world state to a neural network input.

Observation layout (flat float32 vector):
  [0 : GRID_FLAT]   — local NxN terrain window (5 channels)
  [GRID_FLAT : end] — agent state vector

Grid channels per cell (in order):
  0 — terrain type   (normalised to [0, 1])
  1 — soil quality   (normalised to [0, 1])
  2 — tree count     (normalised by max_trees)
  3 — berry count    (normalised by max_berries)
  4 — stone count    (normalised by max_stone)

Agent state vector:
  0   hp             (normalised to [0, 1])
  1   hunger         (normalised to [0, 1])
  2   fatigue        (normalised to [0, 1])
  3   berries carried (log1p normalised)
  4   wood carried   (log1p normalised)
  5   stone carried  (log1p normalised)
  6-9 season         (one-hot, 4 values)
  10  day            (normalised, resets each year = 4 seasons)

Total dimensions: WINDOW*WINDOW*5 + 11

TODO: Add channels for nearby agent positions (requires multi-agent setup).
TODO: Expose memory tokens as extra state dimensions.
TODO: Add visibility / fog-of-war mask.
"""

from __future__ import annotations

import numpy as np

from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim
from ai_world_sim.world.systems.seasons import Season
from ai_world_sim.world.terrain import SoilQuality, TerrainType

NUM_TERRAIN_TYPES = len(TerrainType)
NUM_SEASONS = len(Season)
NUM_GRID_CHANNELS = 5
STATE_DIM = 11
NUM_ACTIONS = 6


def obs_dim(window: int) -> int:
    """Total flat observation size for a given window side length."""
    return window * window * NUM_GRID_CHANNELS + STATE_DIM


def build_observation(
    world: WorldSim,
    agent: Agent,
    window: int = 5,
) -> np.ndarray:
    """Return the flat float32 observation vector for *agent* in *world*."""
    half = window // 2
    r, c = agent.position
    obs = np.zeros((window, window, NUM_GRID_CHANNELS), dtype=np.float32)

    cfg_res = world.config.get("resources", {})
    max_trees = float(cfg_res.get("max_trees", 10) or 1)
    max_berries = float(cfg_res.get("max_berries", 5) or 1)
    max_stone = float(cfg_res.get("max_stone", 8) or 1)

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            wr, wc = dr + half, dc + half
            if not world.in_bounds(nr, nc):
                # Out-of-bounds cells encoded as impassable water with no resources.
                obs[wr, wc, 0] = TerrainType.WATER / (NUM_TERRAIN_TYPES - 1)
                continue
            cell = world.grid[nr][nc]
            obs[wr, wc, 0] = cell.terrain / (NUM_TERRAIN_TYPES - 1)
            obs[wr, wc, 1] = cell.soil / (len(SoilQuality) - 1)
            obs[wr, wc, 2] = cell.trees / max_trees
            obs[wr, wc, 3] = cell.berries / max_berries
            obs[wr, wc, 4] = cell.stone / max_stone

    grid_flat = obs.reshape(-1)

    # Agent state
    agent_cfg = world.config.get("agents", {})
    max_hp = float(agent_cfg.get("max_hp", 100.0))
    max_hunger = float(agent_cfg.get("max_hunger", 100.0))
    max_fatigue = float(agent_cfg.get("max_fatigue", 100.0))

    season_one_hot = np.zeros(NUM_SEASONS, dtype=np.float32)
    season_one_hot[world.season] = 1.0

    year_length = world._season_sys.days_per_season * NUM_SEASONS
    day_norm = (world.day % year_length) / max(year_length, 1)

    state = np.array(
        [
            agent.hp / max_hp,
            agent.hunger / max_hunger,
            agent.fatigue / max_fatigue,
            np.log1p(agent.item_count("berries")) / np.log1p(20),
            np.log1p(agent.item_count("wood")) / np.log1p(20),
            np.log1p(agent.item_count("stone")) / np.log1p(20),
            *season_one_hot,
            day_norm,
        ],
        dtype=np.float32,
    )

    return np.concatenate([grid_flat, state])


def build_action_mask(world: WorldSim, agent: Agent) -> np.ndarray:
    """Return a float32 binary mask of shape (NUM_ACTIONS,).

    1.0 = action is valid, 0.0 = action is blocked.

    Actions:
        0 move_north, 1 move_south, 2 move_east, 3 move_west, 4 forage, 5 rest
    """
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    r, c = agent.position

    # Movement validity
    if not world.is_passable(r - 1, c):
        mask[0] = 0.0  # north
    if not world.is_passable(r + 1, c):
        mask[1] = 0.0  # south
    if not world.is_passable(r, c + 1):
        mask[2] = 0.0  # east
    if not world.is_passable(r, c - 1):
        mask[3] = 0.0  # west

    # Forage only if anything to gather
    if not world.grid[r][c].has_forageable():
        mask[4] = 0.0

    # Rest is always valid (action 5).

    # Safety: at least one action must be valid.
    if mask.sum() == 0.0:
        mask[5] = 1.0  # rest as fallback

    return mask
