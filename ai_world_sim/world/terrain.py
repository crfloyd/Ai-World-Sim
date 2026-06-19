"""Terrain types, soil quality, per-cell data structures, and observation channel map.

Channel indices are defined here so every system that reads or writes the
grid tensor (observation builder, model, tests) uses the same source of truth.

Adding a new channel: append a constant here and update build_grid_tensor()
in observations.py. No other files need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TerrainType(IntEnum):
    GRASS = 0
    FOREST = 1
    MOUNTAIN = 2
    WATER = 3


class SoilQuality(IntEnum):
    POOR = 0
    NORMAL = 1
    FERTILE = 2


# ------------------------------------------------------------------ #
# Observation channel indices (channels × WINDOW × WINDOW tensor)
# ------------------------------------------------------------------ #

# Terrain (one-hot per cell)
CH_TERRAIN_GRASS = 0
CH_TERRAIN_FOREST = 1
CH_TERRAIN_MOUNTAIN = 2
CH_TERRAIN_WATER = 3

# Soil quality (one-hot per cell)
CH_SOIL_POOR = 4
CH_SOIL_NORMAL = 5
CH_SOIL_FERTILE = 6

# Resource amounts (normalised 0–1)
CH_RESOURCE_BERRIES = 7
CH_RESOURCE_TREES = 8
CH_RESOURCE_STONE = 9

# Entity presence (binary)
CH_ENTITY_AGENT = 10      # other agents (future multi-agent)
CH_ENTITY_PREY = 11       # rabbits / deer
CH_ENTITY_PREDATOR = 12   # wolves

# Structures
CH_STRUCTURE_HOME = 13    # agent's own home tile

# Utility
CH_OUT_OF_BOUNDS = 14     # cells outside the world boundary

NUM_CHANNELS = 15


# ------------------------------------------------------------------ #
# Terrain properties
# ------------------------------------------------------------------ #

TERRAIN_PASSABLE: dict[TerrainType, bool] = {
    TerrainType.GRASS: True,
    TerrainType.FOREST: True,
    TerrainType.MOUNTAIN: False,
    TerrainType.WATER: False,
}

TERRAIN_MOVE_COST: dict[TerrainType, float] = {
    TerrainType.GRASS: 1.0,
    TerrainType.FOREST: 1.5,
    TerrainType.MOUNTAIN: float("inf"),
    TerrainType.WATER: float("inf"),
}


@dataclass
class Cell:
    """A single grid cell holding terrain, soil, and resource state."""

    terrain: TerrainType
    soil: SoilQuality

    # Current resource amounts.
    trees: int = 0
    berries: int = 0
    stone: int = 0

    # Carrying capacity (set at generation time, constant thereafter).
    max_trees: int = 0
    max_berries: int = 0
    max_stone: int = 0

    def is_passable(self) -> bool:
        return TERRAIN_PASSABLE[self.terrain]

    def is_water(self) -> bool:
        return self.terrain == TerrainType.WATER

    def has_forageable(self) -> bool:
        """True if the cell has any resource an agent can gather."""
        return self.berries > 0 or self.trees > 0 or self.stone > 0

    def move_cost(self) -> float:
        return TERRAIN_MOVE_COST[self.terrain]
