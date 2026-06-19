"""Engineered memory layer for the agent.

Memory is NOT inside the neural network.  The network receives a compact
numerical summary of memory as part of its observation — it learns to USE
memory, but the storage and update logic is hand-engineered.

This design separates concerns cleanly:
  - What to remember: engineered (food, water, danger locations)
  - How to use memory: learned (policy reads the summary features)

Memory entries decay in confidence over time and are pruned when forgotten.

TODO: Add spatial clustering so nearby known-food cells collapse to one entry.
TODO: Add semantic tags (e.g. "reliable source" if observed multiple times).
TODO: Expose raw memory entries as a sequence for an attention mechanism (V2+).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ai_world_sim.world.terrain import TerrainType

MEMORY_DIM = 12  # size of the summary feature vector


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class MemoryEntry:
    position: tuple[int, int]
    confidence: float       # 1.0 = just seen; decays each tick toward 0
    last_seen_tick: int


class MemoryStore:
    """Stores known locations of food, water, and danger for one agent.

    Updated by the observation builder after each step so the agent always
    has current knowledge of what it can currently see.
    """

    def __init__(self, max_entries: int = 20, decay_rate: float = 0.02) -> None:
        self.max_entries = max_entries
        self.decay_rate = decay_rate

        self.known_food: list[MemoryEntry] = []
        self.known_water: list[MemoryEntry] = []
        self.known_danger: list[MemoryEntry] = []

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        grid: list,          # list[list[Cell]]
        agent_pos: tuple[int, int],
        animals: list,        # list[Animal]
        tick: int,
        sight_radius: int,
        world_height: int,
        world_width: int,
    ) -> None:
        """Scan the visible area and refresh memory entries."""
        r0, c0 = agent_pos
        for dr in range(-sight_radius, sight_radius + 1):
            for dc in range(-sight_radius, sight_radius + 1):
                nr, nc = r0 + dr, c0 + dc
                if not (0 <= nr < world_height and 0 <= nc < world_width):
                    continue
                cell = grid[nr][nc]
                pos = (nr, nc)
                if cell.has_forageable():
                    self._upsert(self.known_food, pos, tick)
                if cell.is_water():
                    self._upsert(self.known_water, pos, tick)

        # Danger: visible wolves
        for animal in animals:
            if not animal.alive or not animal.is_predator:
                continue
            if _manhattan(agent_pos, animal.position) <= sight_radius:
                self._upsert(self.known_danger, animal.position, tick)

    def decay(self, current_tick: int) -> None:
        """Reduce confidence of all entries and prune forgotten ones."""
        for store in (self.known_food, self.known_water, self.known_danger):
            for entry in store:
                age = current_tick - entry.last_seen_tick
                entry.confidence = max(0.0, 1.0 - age * self.decay_rate)
        self.known_food = [e for e in self.known_food if e.confidence > 0.0]
        self.known_water = [e for e in self.known_water if e.confidence > 0.0]
        self.known_danger = [e for e in self.known_danger if e.confidence > 0.0]

    # ------------------------------------------------------------------ #
    # Summary (exposed to policy as observation features)
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        agent_pos: tuple[int, int],
        world_diag: float,
    ) -> np.ndarray:
        """Return a MEMORY_DIM float32 vector summarising memory state.

        Layout:
          [0]   nearest food distance (normalised)
          [1-2] nearest food direction (dr, dc) normalised
          [3]   nearest water distance
          [4-5] nearest water direction
          [6]   nearest danger distance
          [7-8] nearest danger direction
          [9]   num known food locations (log-normalised)
          [10]  num known water locations
          [11]  num known danger locations
        """
        feat = np.zeros(MEMORY_DIM, dtype=np.float32)
        norm = max(world_diag, 1.0)

        self._fill_nearest(feat, 0, self.known_food, agent_pos, norm)
        self._fill_nearest(feat, 3, self.known_water, agent_pos, norm)
        self._fill_nearest(feat, 6, self.known_danger, agent_pos, norm)

        log_max = math.log1p(self.max_entries)
        feat[9] = math.log1p(len(self.known_food)) / log_max
        feat[10] = math.log1p(len(self.known_water)) / log_max
        feat[11] = math.log1p(len(self.known_danger)) / log_max

        return feat

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def nearest_food(self, pos: tuple[int, int]) -> MemoryEntry | None:
        return self._nearest(self.known_food, pos)

    def nearest_water(self, pos: tuple[int, int]) -> MemoryEntry | None:
        return self._nearest(self.known_water, pos)

    def nearest_danger(self, pos: tuple[int, int]) -> MemoryEntry | None:
        return self._nearest(self.known_danger, pos)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _upsert(self, store: list[MemoryEntry], pos: tuple[int, int], tick: int) -> None:
        for entry in store:
            if entry.position == pos:
                entry.confidence = 1.0
                entry.last_seen_tick = tick
                return
        if len(store) < self.max_entries:
            store.append(MemoryEntry(position=pos, confidence=1.0, last_seen_tick=tick))
        else:
            # Replace the least-confident entry.
            oldest = min(store, key=lambda e: e.confidence)
            oldest.position = pos
            oldest.confidence = 1.0
            oldest.last_seen_tick = tick

    @staticmethod
    def _nearest(store: list[MemoryEntry], pos: tuple[int, int]) -> MemoryEntry | None:
        if not store:
            return None
        return min(store, key=lambda e: _manhattan(pos, e.position))

    @staticmethod
    def _fill_nearest(
        feat: np.ndarray,
        offset: int,
        store: list[MemoryEntry],
        pos: tuple[int, int],
        norm: float,
    ) -> None:
        entry = MemoryStore._nearest(store, pos)
        if entry is None:
            return
        dist = _manhattan(pos, entry.position)
        dr = (entry.position[0] - pos[0]) / (dist + 1e-8)
        dc = (entry.position[1] - pos[1]) / (dist + 1e-8)
        feat[offset] = min(1.0, dist / norm)
        feat[offset + 1] = float(np.clip(dr, -1.0, 1.0))
        feat[offset + 2] = float(np.clip(dc, -1.0, 1.0))
