"""Season tracking and progression.

The season advances automatically as the world's day counter increments.
Seasons affect resource regeneration rates, hunger/fatigue multipliers,
and (in the future) migration pressures and weather events.

TODO: Add weather events (drought, blizzard) that modify resource regen.
TODO: Drive NPC migration triggers from season transitions.
"""

from __future__ import annotations

from enum import IntEnum


class Season(IntEnum):
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3

    def next(self) -> "Season":
        return Season((self.value + 1) % 4)

    def label(self) -> str:
        return self.name.capitalize()


class SeasonSystem:
    """Determines the current season from the world's day counter."""

    def __init__(self, days_per_season: int = 30) -> None:
        self.days_per_season = days_per_season

    def season_for_day(self, day: int) -> Season:
        """Return the season active on the given absolute day number."""
        season_index = (day // self.days_per_season) % 4
        return Season(season_index)

    def day_within_season(self, day: int) -> int:
        """How many days into the current season we are (0-indexed)."""
        return day % self.days_per_season

    def days_until_next_season(self, day: int) -> int:
        return self.days_per_season - self.day_within_season(day)

    def did_season_change(self, old_day: int, new_day: int) -> bool:
        return self.season_for_day(old_day) != self.season_for_day(new_day)
