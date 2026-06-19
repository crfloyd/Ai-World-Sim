"""Tests for the MemoryStore engineered memory layer."""

from __future__ import annotations

import pytest

from ai_world_sim.world.memory import MemoryStore


def test_memory_starts_empty():
    m = MemoryStore()
    assert m.known_food == []
    assert m.known_water == []
    assert m.known_danger == []


def test_upsert_adds_entry():
    m = MemoryStore()
    m._upsert(m.known_food, (3, 4), tick=0)
    assert len(m.known_food) == 1
    assert m.known_food[0].position == (3, 4)
    assert m.known_food[0].confidence == 1.0


def test_upsert_same_position_refreshes():
    m = MemoryStore()
    m._upsert(m.known_food, (3, 4), tick=0)
    m.known_food[0].confidence = 0.5
    m._upsert(m.known_food, (3, 4), tick=10)
    assert len(m.known_food) == 1
    assert m.known_food[0].confidence == 1.0
    assert m.known_food[0].last_seen_tick == 10


def test_decay_reduces_confidence():
    m = MemoryStore(decay_rate=0.1)
    m._upsert(m.known_food, (1, 1), tick=0)
    m.decay(current_tick=5)
    # confidence = 1.0 - 5*0.1 = 0.5
    assert abs(m.known_food[0].confidence - 0.5) < 1e-6


def test_decay_removes_forgotten_entries():
    m = MemoryStore(decay_rate=0.1)
    m._upsert(m.known_food, (1, 1), tick=0)
    m.decay(current_tick=11)  # confidence = 1.0 - 11*0.1 < 0 → pruned
    assert m.known_food == []


def test_nearest_returns_closest():
    m = MemoryStore()
    m._upsert(m.known_food, (0, 10), tick=0)
    m._upsert(m.known_food, (0, 3), tick=0)
    m._upsert(m.known_food, (0, 7), tick=0)
    nearest = m.nearest_food((0, 0))
    assert nearest.position == (0, 3)


def test_nearest_returns_none_when_empty():
    m = MemoryStore()
    assert m.nearest_food((0, 0)) is None


def test_summarize_returns_correct_shape():
    m = MemoryStore()
    feat = m.summarize(agent_pos=(5, 5), world_diag=100.0)
    from ai_world_sim.world.memory import MEMORY_DIM
    assert feat.shape == (MEMORY_DIM,)


def test_summarize_zero_when_no_memory():
    m = MemoryStore()
    feat = m.summarize(agent_pos=(5, 5), world_diag=100.0)
    # Distance slots (0, 3, 6) should be 0 when no entries.
    assert feat[0] == 0.0
    assert feat[3] == 0.0
    assert feat[6] == 0.0


def test_summarize_food_distance():
    m = MemoryStore()
    m._upsert(m.known_food, (5, 15), tick=0)  # manhattan dist = 10 from (5,5)
    feat = m.summarize(agent_pos=(5, 5), world_diag=100.0)
    expected_dist = 10.0 / 100.0
    assert abs(feat[0] - expected_dist) < 1e-5


def test_update_adds_food_from_visible_area(tmp_path):
    """Stub a minimal world-like grid and verify update() picks up food."""
    from ai_world_sim.world.terrain import Cell, TerrainType, SoilQuality

    # Build a 10x10 grid with berries at (5, 5).
    grid = [
        [Cell(terrain=TerrainType.GRASS, soil=SoilQuality.NORMAL) for _ in range(10)]
        for _ in range(10)
    ]
    grid[5][5].berries = 3
    grid[5][5].max_berries = 5

    m = MemoryStore()
    m.update(
        grid=grid,
        agent_pos=(5, 5),
        animals=[],
        tick=0,
        sight_radius=3,
        world_height=10,
        world_width=10,
    )
    assert any(e.position == (5, 5) for e in m.known_food)


def test_update_adds_water_from_visible_area():
    from ai_world_sim.world.terrain import Cell, TerrainType, SoilQuality

    grid = [
        [Cell(terrain=TerrainType.GRASS, soil=SoilQuality.NORMAL) for _ in range(10)]
        for _ in range(10)
    ]
    grid[4][4].terrain = TerrainType.WATER

    m = MemoryStore()
    m.update(
        grid=grid,
        agent_pos=(5, 5),
        animals=[],
        tick=0,
        sight_radius=3,
        world_height=10,
        world_width=10,
    )
    assert any(e.position == (4, 4) for e in m.known_water)


def test_max_entries_cap():
    m = MemoryStore(max_entries=3)
    for i in range(10):
        m._upsert(m.known_food, (0, i), tick=i)
    assert len(m.known_food) <= 3
