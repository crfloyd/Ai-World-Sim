"""Agent entity definition.

Each agent is an instance of the shared neural policy operating inside the world.
Traits, skills, and memory are deliberately kept as open dicts so the observation
encoder can expose whichever subset the policy sees — and so future systems
(relationships, factions, culture) can attach data without changing this class.

TODO: Add relationship map (trust/hostility per other agent id).
TODO: Add faction membership.
TODO: Add long-term episodic memory structures.
TODO: Add animal entities (prey, predators).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    """A single individual instantiated from the shared policy.

    All numeric stats are floats so the observation encoder can normalise them
    without casting.
    """

    id: int
    position: tuple[int, int]  # (row, col)

    hp: float = 100.0
    hunger: float = 0.0
    fatigue: float = 0.0

    # Counts of carried items: {"berries": 3, "wood": 1, "stone": 0, ...}
    inventory: dict[str, int] = field(default_factory=dict)

    # Continuous scalar traits, e.g. {"strength": 0.8, "agility": 0.6}.
    # Values are sampled at spawn and remain fixed (nature, not learned).
    traits: dict[str, float] = field(default_factory=dict)

    # Learned skill levels, e.g. {"foraging": 0.2, "crafting": 0.0}.
    # TODO: increment skills on successful action use.
    skills: dict[str, float] = field(default_factory=dict)

    # Short-term working memory written by the brain or systems.
    # e.g. {"last_food_pos": (12, 7), "seen_danger": False}
    # TODO: Replace with a structured episodic memory buffer.
    memory: dict[str, Any] = field(default_factory=dict)

    alive: bool = True
    age: int = 0  # ticks lived

    # ------------------------------------------------------------------ #
    # Inventory helpers
    # ------------------------------------------------------------------ #

    def add_item(self, item: str, amount: int = 1) -> None:
        self.inventory[item] = self.inventory.get(item, 0) + amount

    def remove_item(self, item: str, amount: int = 1) -> bool:
        """Remove *amount* of *item*. Returns False if insufficient stock."""
        if self.inventory.get(item, 0) >= amount:
            self.inventory[item] -= amount
            if self.inventory[item] == 0:
                del self.inventory[item]
            return True
        return False

    def item_count(self, item: str) -> int:
        return self.inventory.get(item, 0)

    def total_carried(self) -> int:
        return sum(self.inventory.values())

    # ------------------------------------------------------------------ #
    # State queries
    # ------------------------------------------------------------------ #

    def is_starving(self) -> bool:
        return self.hunger >= 100.0

    def is_exhausted(self) -> bool:
        return self.fatigue >= 100.0

    def is_dead(self) -> bool:
        return not self.alive or self.hp <= 0.0
