"""Procedural world generation from a seeded RNG.

The algorithm:
  1. Generate a raw heightmap with numpy random.
  2. Smooth it with a simple box filter to create coherent regions.
  3. Threshold into terrain types (water → grass → forest → mountain).
  4. Generate a separate noise map for soil quality.
  5. Place resources according to terrain type and density config.

The same seed always produces the same world (deterministic).

TODO: Replace box-filter smoothing with Perlin / simplex noise for
      more realistic continent shapes.
TODO: Add river generation (connected water paths from mountains to coast).
TODO: Add biome tags beyond terrain type (desert, tundra, wetlands).
"""

from __future__ import annotations

import numpy as np

from ai_world_sim.world.terrain import Cell, SoilQuality, TerrainType


def _smooth(arr: np.ndarray, passes: int = 4) -> np.ndarray:
    """Apply a simple 3×3 box-filter blur *passes* times."""
    result = arr.copy().astype(np.float64)
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, :-2] + padded[1:-1, :-2] + padded[2:, :-2]
            + padded[:-2, 1:-1] + padded[1:-1, 1:-1] + padded[2:, 1:-1]
            + padded[:-2, 2:] + padded[1:-1, 2:] + padded[2:, 2:]
        ) / 9.0
    # Renormalize to [0, 1].
    lo, hi = result.min(), result.max()
    if hi > lo:
        result = (result - lo) / (hi - lo)
    return result


def generate_world(
    width: int,
    height: int,
    seed: int,
    config: dict,
) -> list[list[Cell]]:
    """Return a *height* × *width* grid of :class:`Cell` objects.

    Parameters
    ----------
    width, height:
        Grid dimensions.
    seed:
        Integer seed for reproducible generation.
    config:
        Full world config dict (see ``configs/world.yaml``).
    """
    rng = np.random.default_rng(seed)
    terrain_cfg = config.get("terrain", {})
    res_cfg = config.get("resources", {})

    water_t: float = terrain_cfg.get("water_threshold", 0.25)
    grass_t: float = terrain_cfg.get("grass_threshold", 0.55)
    forest_t: float = terrain_cfg.get("forest_threshold", 0.75)

    tree_density: float = res_cfg.get("tree_density", 0.6)
    berry_density: float = res_cfg.get("berry_density", 0.4)
    stone_density: float = res_cfg.get("stone_density", 0.2)
    max_trees: int = int(res_cfg.get("max_trees", 10))
    max_berries: int = int(res_cfg.get("max_berries", 5))
    max_stone: int = int(res_cfg.get("max_stone", 8))

    # --- Heightmap ------------------------------------------------------ #
    raw_height = rng.random((height, width))
    height_map = _smooth(raw_height, passes=4)

    # --- Soil quality map ----------------------------------------------- #
    raw_soil = rng.random((height, width))
    soil_map = _smooth(raw_soil, passes=2)

    # --- Build grid ---------------------------------------------------- #
    grid: list[list[Cell]] = []
    for r in range(height):
        row: list[Cell] = []
        for c in range(width):
            h = height_map[r, c]
            s = soil_map[r, c]

            # Terrain
            if h < water_t:
                terrain = TerrainType.WATER
            elif h < grass_t:
                terrain = TerrainType.GRASS
            elif h < forest_t:
                terrain = TerrainType.FOREST
            else:
                terrain = TerrainType.MOUNTAIN

            # Soil quality
            if s < 0.33:
                soil = SoilQuality.POOR
            elif s < 0.67:
                soil = SoilQuality.NORMAL
            else:
                soil = SoilQuality.FERTILE

            # Resources (only on passable cells)
            trees = berries = stone = 0
            m_trees = m_berries = m_stone = 0

            if terrain == TerrainType.FOREST:
                if rng.random() < tree_density:
                    m_trees = max_trees
                    trees = int(rng.integers(max_trees // 2, max_trees + 1))
                if rng.random() < berry_density * 0.5:
                    m_berries = max_berries
                    berries = int(rng.integers(0, max_berries + 1))

            elif terrain == TerrainType.GRASS:
                if rng.random() < tree_density * 0.3:
                    m_trees = max_trees // 2
                    trees = int(rng.integers(0, m_trees + 1))
                if rng.random() < berry_density:
                    m_berries = max_berries
                    berries = int(rng.integers(max_berries // 2, max_berries + 1))

            # Stone appears on grass and forest floors.
            if terrain in (TerrainType.GRASS, TerrainType.FOREST):
                if rng.random() < stone_density:
                    m_stone = max_stone
                    stone = int(rng.integers(max_stone // 2, max_stone + 1))

            row.append(
                Cell(
                    terrain=terrain,
                    soil=soil,
                    trees=trees,
                    berries=berries,
                    stone=stone,
                    max_trees=m_trees,
                    max_berries=m_berries,
                    max_stone=m_stone,
                )
            )
        grid.append(row)

    return grid


def find_spawn_positions(
    grid: list[list[Cell]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return *n* passable cell positions for agent spawning."""
    passable = [
        (r, c)
        for r, row in enumerate(grid)
        for c, cell in enumerate(row)
        if cell.is_passable()
    ]
    if len(passable) < n:
        raise RuntimeError(
            f"Not enough passable cells ({len(passable)}) to spawn {n} agents."
        )
    indices = rng.choice(len(passable), size=n, replace=False)
    return [passable[i] for i in indices]
