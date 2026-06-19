"""Agent health, hunger, thirst, and fatigue tick processing.

Applied once per tick to every living agent by WorldSim.
All thresholds are read from the world config so they can be tuned
without touching code.

Stat priority (speed of danger):
  thirst → fastest to critical
  hunger → medium
  tired  → slow but compounds other stats

TODO: Add disease / poison status effects.
TODO: Add temperature stress (cold in winter without shelter or fire).
TODO: Age-based stat degradation for long-lived agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_world_sim.world.entities import Agent
    from ai_world_sim.story.events import EventLog


class HealthSystem:
    """Updates survival stats for an agent each tick."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("agents", {})
        self.hunger_per_tick: float = cfg.get("hunger_per_tick", 0.3)
        self.thirst_per_tick: float = cfg.get("thirst_per_tick", 0.5)
        self.tired_per_tick: float = cfg.get("tired_per_tick", 0.2)

        self.starvation_hp_loss: float = cfg.get("starvation_hp_loss", 2.0)
        self.dehydration_hp_loss: float = cfg.get("dehydration_hp_loss", 3.0)

        self.max_hp: float = cfg.get("max_hp", 100.0)
        self.max_hunger: float = cfg.get("max_hunger", 100.0)
        self.max_thirst: float = cfg.get("max_thirst", 100.0)
        self.max_tired: float = cfg.get("max_tired", 100.0)

        # Rest
        self.rest_tired_recovery: float = cfg.get("rest_tired_recovery", 5.0)
        self.rest_hp_recovery: float = cfg.get("rest_hp_recovery", 0.5)

        # Sleep
        self.sleep_tired_recovery: float = cfg.get("sleep_tired_recovery", 20.0)
        self.sleep_hp_recovery: float = cfg.get("sleep_hp_recovery", 2.0)
        self.sleep_home_tired_bonus: float = cfg.get("sleep_home_tired_bonus", 10.0)
        self.sleep_home_hp_bonus: float = cfg.get("sleep_home_hp_bonus", 1.0)

        # Food / drink
        self.eat_hunger_reduction: float = cfg.get("eat_hunger_reduction", 25.0)
        self.eat_meat_hunger_reduction: float = cfg.get("eat_meat_hunger_reduction", 40.0)
        self.drink_thirst_reduction: float = cfg.get("drink_thirst_reduction", 40.0)

    def tick(self, agent: "Agent", event_log: "EventLog | None" = None) -> None:
        """Mutate *agent* stats for one tick. Logs death if it occurs."""
        if not agent.alive:
            return

        agent.hunger = min(self.max_hunger, agent.hunger + self.hunger_per_tick)
        agent.thirst = min(self.max_thirst, agent.thirst + self.thirst_per_tick)
        agent.tired = min(self.max_tired, agent.tired + self.tired_per_tick)
        agent.age += 1

        # Starvation damage
        if agent.is_starving():
            agent.hp = max(0.0, agent.hp - self.starvation_hp_loss)

        # Dehydration damage (happens faster than starvation)
        if agent.is_dehydrated():
            agent.hp = max(0.0, agent.hp - self.dehydration_hp_loss)

        if agent.hp <= 0.0:
            agent.alive = False
            if event_log is not None:
                cause = "dehydration" if agent.is_dehydrated() else "starvation"
                event_log.log(f"Agent {agent.id} died of {cause}.")

    # ------------------------------------------------------------------ #
    # Action effects
    # ------------------------------------------------------------------ #

    def apply_eat_berries(self, agent: "Agent") -> None:
        agent.hunger = max(0.0, agent.hunger - self.eat_hunger_reduction)

    def apply_eat_meat(self, agent: "Agent") -> None:
        agent.hunger = max(0.0, agent.hunger - self.eat_meat_hunger_reduction)

    def apply_drink(self, agent: "Agent") -> None:
        agent.thirst = max(0.0, agent.thirst - self.drink_thirst_reduction)

    def apply_rest(self, agent: "Agent") -> None:
        agent.tired = max(0.0, agent.tired - self.rest_tired_recovery)
        agent.hp = min(self.max_hp, agent.hp + self.rest_hp_recovery)

    def apply_sleep(self, agent: "Agent") -> None:
        tired_recovery = self.sleep_tired_recovery
        hp_recovery = self.sleep_hp_recovery
        if agent.is_at_home:
            tired_recovery += self.sleep_home_tired_bonus
            hp_recovery += self.sleep_home_hp_bonus
        agent.tired = max(0.0, agent.tired - tired_recovery)
        agent.hp = min(self.max_hp, agent.hp + hp_recovery)

    def apply_tired_move_cost(self, agent: "Agent", cost: float) -> None:
        """Extra tired cost for movement actions."""
        agent.tired = min(self.max_tired, agent.tired + cost)
