"""Agent health, hunger, and fatigue tick processing.

Applied once per tick to every living agent by WorldSim.
All thresholds are read from the world config so they can be tuned
without touching code.

TODO: Add disease/poison status effects.
TODO: Add temperature stress (cold in winter without shelter/fire).
TODO: Age-based stat degradation for long-lived agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_world_sim.world.entities import Agent
    from ai_world_sim.story.events import EventLog


class HealthSystem:
    """Updates hunger, fatigue, and HP for an agent each tick."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("agents", {})
        self.hunger_per_tick: float = cfg.get("hunger_per_tick", 0.5)
        self.fatigue_per_tick: float = cfg.get("fatigue_per_tick", 0.3)
        self.starvation_hp_loss: float = cfg.get("starvation_hp_loss", 2.0)
        self.max_hunger: float = cfg.get("max_hunger", 100.0)
        self.max_fatigue: float = cfg.get("max_fatigue", 100.0)
        self.max_hp: float = cfg.get("max_hp", 100.0)

    def tick(self, agent: "Agent", event_log: "EventLog | None" = None) -> None:
        """Mutate *agent* stats for one tick. Logs death if it occurs."""
        if not agent.alive:
            return

        agent.hunger = min(self.max_hunger, agent.hunger + self.hunger_per_tick)
        agent.fatigue = min(self.max_fatigue, agent.fatigue + self.fatigue_per_tick)
        agent.age += 1

        if agent.is_starving():
            agent.hp = max(0.0, agent.hp - self.starvation_hp_loss)

        if agent.hp <= 0.0:
            agent.alive = False
            if event_log is not None:
                event_log.log(f"Agent {agent.id} died of starvation.")

    def apply_eat(self, agent: "Agent", amount: int) -> None:
        """Reduce hunger when the agent consumes food."""
        hunger_reduction = amount * 20.0
        agent.hunger = max(0.0, agent.hunger - hunger_reduction)

    def apply_rest(self, agent: "Agent", recovery: float) -> None:
        """Reduce fatigue when the agent rests."""
        agent.fatigue = max(0.0, agent.fatigue - recovery)
