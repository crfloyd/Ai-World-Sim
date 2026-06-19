"""Event logging framework.

Every meaningful action in the simulation should route through EventLog.log()
so that stories, statistics, and debugging all draw from the same source.

Events are stored as (tick, day, season_label, message) tuples in memory.
Future versions can persist to SQLite, stream to a story model, or feed into
an LLM narrator.

TODO: Add structured event types (dataclasses) beyond raw strings for easier
      querying (e.g. "show all deaths", "show all trades").
TODO: Persist events per episode to disk for offline story generation.
TODO: Feed event stream to a language model to generate narrative text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Event:
    tick: int
    day: int
    season: str
    message: str

    def __str__(self) -> str:
        return f"[Day {self.day} / {self.season} / tick {self.tick}] {self.message}"


class EventLog:
    """Collects and stores simulation events for a single episode."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._tick: int = 0
        self._day: int = 0
        self._season: str = "Spring"

    def update_time(self, tick: int, day: int, season: str) -> None:
        """Called by WorldSim each tick so events are time-stamped correctly."""
        self._tick = tick
        self._day = day
        self._season = season

    def log(self, message: str) -> None:
        self._events.append(
            Event(
                tick=self._tick,
                day=self._day,
                season=self._season,
                message=message,
            )
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def since_tick(self, tick: int) -> list[Event]:
        return [e for e in self._events if e.tick >= tick]

    def filter_containing(self, keyword: str) -> list[Event]:
        return [e for e in self._events if keyword.lower() in e.message.lower()]

    def last(self, n: int = 10) -> list[Event]:
        return self._events[-n:]

    def __len__(self) -> int:
        return len(self._events)

    def __str__(self) -> str:
        return "\n".join(str(e) for e in self._events)
