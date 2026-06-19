"""Resource regeneration system.

Resources regenerate passively each tick at rates determined by the
current season. The regeneration is applied to the world's grid of Cell
objects so the effect is immediately visible to all agents.

TODO: Soil quality should modulate regen rate for berry/tree cells.
TODO: Agent farming actions should temporarily boost regen on worked cells.
TODO: Overcrowding (too many agents harvesting a cell) should suppress regen.
"""

from __future__ import annotations

import numpy as np

from ai_world_sim.world.systems.seasons import Season
from ai_world_sim.world.terrain import Cell, TerrainType


# Seasonal regen rate config keys
_REGEN_KEYS: dict[Season, str] = {
    Season.SPRING: "regen_rate_spring",
    Season.SUMMER: "regen_rate_summer",
    Season.AUTUMN: "regen_rate_autumn",
    Season.WINTER: "regen_rate_winter",
}


class ResourceSystem:
    """Handles passive regeneration of trees, berries, and stone each tick."""

    def __init__(self, config: dict, rng: np.random.Generator) -> None:
        self.cfg = config.get("resources", {})
        self.rng = rng

    def _regen_rate(self, season: Season) -> float:
        key = _REGEN_KEYS[season]
        return float(self.cfg.get(key, 0.0))

    def tick(self, grid: list[list[Cell]], season: Season) -> None:
        """Probabilistically restore resources across all cells."""
        rate = self._regen_rate(season)
        if rate <= 0.0:
            return

        for row in grid:
            for cell in row:
                if not cell.is_passable():
                    continue
                # Each resource type regenerates independently at *rate* per tick.
                if cell.berries < cell.max_berries and self.rng.random() < rate:
                    cell.berries = min(cell.berries + 1, cell.max_berries)
                if cell.trees < cell.max_trees and self.rng.random() < rate:
                    cell.trees = min(cell.trees + 1, cell.max_trees)
                # Stone regenerates very slowly (geological timescale stub).
                if cell.stone < cell.max_stone and self.rng.random() < rate * 0.1:
                    cell.stone = min(cell.stone + 1, cell.max_stone)
