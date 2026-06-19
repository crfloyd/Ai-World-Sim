"""Tests for the season system."""

from __future__ import annotations

import pytest

from ai_world_sim.world.systems.seasons import Season, SeasonSystem


@pytest.fixture()
def sys30():
    return SeasonSystem(days_per_season=30)


def test_spring_is_first(sys30):
    assert sys30.season_for_day(0) == Season.SPRING


def test_summer_starts_at_day_30(sys30):
    assert sys30.season_for_day(30) == Season.SUMMER


def test_autumn_starts_at_day_60(sys30):
    assert sys30.season_for_day(60) == Season.AUTUMN


def test_winter_starts_at_day_90(sys30):
    assert sys30.season_for_day(90) == Season.WINTER


def test_season_cycles_back_to_spring(sys30):
    assert sys30.season_for_day(120) == Season.SPRING


def test_season_cycle_long(sys30):
    """Two full years should cycle correctly."""
    for year in range(2):
        offset = year * 120
        assert sys30.season_for_day(offset + 0) == Season.SPRING
        assert sys30.season_for_day(offset + 30) == Season.SUMMER
        assert sys30.season_for_day(offset + 60) == Season.AUTUMN
        assert sys30.season_for_day(offset + 90) == Season.WINTER


def test_day_within_season(sys30):
    assert sys30.day_within_season(0) == 0
    assert sys30.day_within_season(15) == 15
    assert sys30.day_within_season(30) == 0
    assert sys30.day_within_season(45) == 15


def test_days_until_next_season(sys30):
    assert sys30.days_until_next_season(0) == 30
    assert sys30.days_until_next_season(15) == 15
    assert sys30.days_until_next_season(29) == 1


def test_did_season_change(sys30):
    assert sys30.did_season_change(29, 30) is True
    assert sys30.did_season_change(28, 29) is False


def test_season_next():
    assert Season.SPRING.next() == Season.SUMMER
    assert Season.SUMMER.next() == Season.AUTUMN
    assert Season.AUTUMN.next() == Season.WINTER
    assert Season.WINTER.next() == Season.SPRING
