"""Terrain types, soil quality, and per-cell data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
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


# Terrain passability lookup — agents cannot enter impassable cells.
TERRAIN_PASSABLE: dict[TerrainType, bool] = {
    TerrainType.GRASS: True,
    TerrainType.FOREST: True,
    TerrainType.MOUNTAIN: False,
    TerrainType.WATER: False,
}

# Move cost multiplier (future: fatigue scales with terrain difficulty).
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

    # Maximum carrying capacity (set at generation time).
    max_trees: int = 0
    max_berries: int = 0
    max_stone: int = 0

    def is_passable(self) -> bool:
        return TERRAIN_PASSABLE[self.terrain]

    def has_forageable(self) -> bool:
        """Returns True if the cell has any resource an agent can gather."""
        return self.berries > 0 or self.trees > 0 or self.stone > 0

    def move_cost(self) -> float:
        return TERRAIN_MOVE_COST[self.terrain]
