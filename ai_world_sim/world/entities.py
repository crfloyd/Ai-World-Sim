"""Agent entity definition.

Each agent is an instance of the shared neural policy operating inside the world.
Traits, skills, and memory are deliberately kept extensible so future systems
(relationships, factions, culture) can attach data without changing this class.

All survival stats are normalised to [0, 1] in the observation builder —
raw values are stored here at their natural scale.

TODO: Add relationship map (trust/hostility per other agent id).
TODO: Add faction membership and reputation scores.
TODO: Add long-term episodic memory structures beyond MemoryStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_world_sim.world.memory import MemoryStore


@dataclass
class Agent:
    """A single individual instantiated from the shared policy."""

    id: int
    position: tuple[int, int]       # (row, col)
    home_position: tuple[int, int]  # spawn location; used for home storage / safe sleep

    # Survival stats
    hp: float = 100.0
    hunger: float = 0.0
    thirst: float = 0.0
    tired: float = 0.0

    # Inventory: items carried on the agent
    inventory: dict[str, int] = field(default_factory=dict)

    # Home storage: food cached at home_position
    stored_food: dict[str, int] = field(default_factory=dict)

    # Intrinsic traits (fixed at spawn; nature not learned)
    traits: dict[str, float] = field(default_factory=dict)

    # Learned skill levels — incremented by systems on successful use
    # TODO: increment skills on repeated successful use.
    skills: dict[str, float] = field(default_factory=dict)

    # Short-term working memory exposed to systems
    # The MemoryStore handles geographic memory; this dict is for ad-hoc state.
    memory: MemoryStore = field(default_factory=MemoryStore)

    # Status flags
    alive: bool = True
    sleeping: bool = False  # True while sleep action is active; wolves deal bonus damage
    age: int = 0            # ticks lived

    # ------------------------------------------------------------------ #
    # Derived state queries
    # ------------------------------------------------------------------ #

    @property
    def is_at_home(self) -> bool:
        return self.position == self.home_position

    def is_starving(self) -> bool:
        return self.hunger >= 100.0

    def is_dehydrated(self) -> bool:
        return self.thirst >= 100.0

    def is_exhausted(self) -> bool:
        return self.tired >= 100.0

    def is_dead(self) -> bool:
        return not self.alive or self.hp <= 0.0

    # ------------------------------------------------------------------ #
    # Inventory helpers
    # ------------------------------------------------------------------ #

    def add_item(self, item: str, amount: int = 1) -> None:
        self.inventory[item] = self.inventory.get(item, 0) + amount

    def remove_item(self, item: str, amount: int = 1) -> bool:
        """Remove *amount* of *item* from inventory. Returns False if insufficient."""
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

    def has_food(self) -> bool:
        """True if the agent carries any edible item."""
        return self.item_count("berries") > 0 or self.item_count("meat") > 0

    def has_stored_food(self) -> bool:
        return sum(self.stored_food.values()) > 0

    # ------------------------------------------------------------------ #
    # Home storage helpers
    # ------------------------------------------------------------------ #

    def store_all_food(self) -> int:
        """Move all food from inventory to stored_food. Returns units stored."""
        total = 0
        for item in ("berries", "meat"):
            count = self.inventory.pop(item, 0)
            if count:
                self.stored_food[item] = self.stored_food.get(item, 0) + count
                total += count
        return total

    def retrieve_all_food(self) -> int:
        """Move all stored food to inventory. Returns units retrieved."""
        total = 0
        for item in list(self.stored_food.keys()):
            count = self.stored_food.pop(item, 0)
            if count:
                self.inventory[item] = self.inventory.get(item, 0) + count
                total += count
        return total

    def retrieve_up_to(self, max_units: int) -> int:
        """Move up to *max_units* of stored food to inventory.

        Prioritises berries then meat.  Returns the number of units retrieved.
        """
        remaining = max_units
        total = 0
        for item in ("berries", "meat"):
            if remaining <= 0:
                break
            available = self.stored_food.get(item, 0)
            if available > 0:
                take = min(available, remaining)
                self.stored_food[item] -= take
                if self.stored_food[item] == 0:
                    del self.stored_food[item]
                self.inventory[item] = self.inventory.get(item, 0) + take
                total += take
                remaining -= take
        return total
