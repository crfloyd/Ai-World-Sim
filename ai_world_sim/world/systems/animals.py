"""Scripted animal behavior system.

Animals use rule-based logic in V0 — NOT neural policies.
They are simulated each tick by AnimalSystem.tick().

Behavior summary:
  Rabbit / Deer (prey):
    - wander randomly
    - flee if agent or wolf within flee_range

  Wolf (predator):
    - wander randomly
    - if agent within hunt_range:
        - move toward agent
        - attack if adjacent (bonus damage on sleeping agents outside home)
    - elif prey within hunt_range:
        - move toward prey, kill if adjacent

The scripted logic is intentionally simple; the point is ecological pressure
not realistic animal AI.

TODO: V1+ — replace scripted wolves with a separate trained policy.
TODO: V1+ — add reproduction and population homeostasis.
TODO: V1+ — add territorial range limits per animal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ai_world_sim.world.animals import Animal, AnimalSpecies

if TYPE_CHECKING:
    from ai_world_sim.story.events import EventLog
    from ai_world_sim.world.entities import Agent
    from ai_world_sim.world.sim import WorldSim


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(
    pos: tuple[int, int],
    target: tuple[int, int],
    world: "WorldSim",
) -> tuple[int, int]:
    """Move one cell toward *target* along the dominant axis."""
    r, c = pos
    tr, tc = target
    candidates: list[tuple[int, int]] = []
    if tr != r:
        candidates.append((r + (1 if tr > r else -1), c))
    if tc != c:
        candidates.append((r, c + (1 if tc > c else -1)))
    for nr, nc in candidates:
        if world.is_passable(nr, nc):
            return (nr, nc)
    return pos


def _flee_from(
    pos: tuple[int, int],
    threat: tuple[int, int],
    world: "WorldSim",
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Move one cell away from *threat*."""
    r, c = pos
    tr, tc = threat
    candidates: list[tuple[int, int]] = []
    if tr != r:
        candidates.append((r + (-1 if tr > r else 1), c))
    if tc != c:
        candidates.append((r, c + (-1 if tc > c else 1)))
    for nr, nc in candidates:
        if world.is_passable(nr, nc):
            return (nr, nc)
    return _random_move(pos, world, rng)


def _random_move(
    pos: tuple[int, int],
    world: "WorldSim",
    rng: np.random.Generator,
) -> tuple[int, int]:
    r, c = pos
    options = [
        (r + dr, c + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1))
        if world.is_passable(r + dr, c + dc)
    ]
    if not options:
        return pos
    return options[int(rng.integers(len(options)))]


class AnimalSystem:
    """Runs one tick of scripted behavior for all animals."""

    def __init__(self, config: dict, rng: np.random.Generator) -> None:
        anim_cfg = config.get("animals", {})
        self.wolves_enabled: bool = bool(anim_cfg.get("wolves_enabled", True))
        self.flee_range: int = int(anim_cfg.get("flee_range", 5))
        # wolf_hunt_range takes precedence over the legacy hunt_range key
        self.hunt_range: int = int(
            anim_cfg.get("wolf_hunt_range", anim_cfg.get("hunt_range", 10))
        )
        self.wolf_attack_damage: float = float(anim_cfg.get("wolf_attack_damage", 15.0))
        self.wolf_sleep_bonus: float = float(anim_cfg.get("wolf_sleep_bonus_damage", 10.0))
        self.rng = rng

    def tick(
        self,
        world: "WorldSim",
        animals: list[Animal],
        agents: list["Agent"],
        event_log: "EventLog | None" = None,
    ) -> None:
        wolves = [a for a in animals if a.is_predator and a.alive]
        prey_list = [a for a in animals if a.is_prey and a.alive]
        living_agents = [a for a in agents if a.alive]

        for animal in animals:
            if not animal.alive:
                continue
            if animal.is_predator:
                if self.wolves_enabled:
                    self._update_wolf(animal, world, living_agents, prey_list, event_log)
            else:
                self._update_prey(animal, world, living_agents, wolves)

    def _update_wolf(
        self,
        wolf: Animal,
        world: "WorldSim",
        agents: list["Agent"],
        prey_list: list[Animal],
        event_log: "EventLog | None",
    ) -> None:
        # 1. Look for nearest agent within hunt_range.
        nearest_agent: "Agent | None" = None
        nearest_agent_dist = float("inf")
        for agent in agents:
            d = _manhattan(wolf.position, agent.position)
            if d <= self.hunt_range and d < nearest_agent_dist:
                nearest_agent_dist = d
                nearest_agent = agent

        if nearest_agent is not None:
            if nearest_agent_dist <= 1:
                # Attack — but wolves respect home safety.
                if not nearest_agent.is_at_home:
                    damage = self.wolf_attack_damage
                    if nearest_agent.sleeping:
                        damage += self.wolf_sleep_bonus
                    nearest_agent.hp = max(0.0, nearest_agent.hp - damage)
                    if event_log:
                        event_log.log(
                            f"Wolf {wolf.id} attacked Agent {nearest_agent.id} "
                            f"for {damage:.0f} damage."
                        )
                    if nearest_agent.hp <= 0.0:
                        nearest_agent.alive = False
                        if event_log:
                            event_log.log(f"Agent {nearest_agent.id} was killed by Wolf {wolf.id}.")
            else:
                wolf.position = _step_toward(wolf.position, nearest_agent.position, world)
            return

        # 2. Look for nearest prey.
        nearest_prey: Animal | None = None
        nearest_prey_dist = float("inf")
        for prey in prey_list:
            d = _manhattan(wolf.position, prey.position)
            if d <= self.hunt_range and d < nearest_prey_dist:
                nearest_prey_dist = d
                nearest_prey = prey

        if nearest_prey is not None:
            if nearest_prey_dist <= 1:
                nearest_prey.alive = False  # wolf kills prey silently
            else:
                wolf.position = _step_toward(wolf.position, nearest_prey.position, world)
            return

        wolf.position = _random_move(wolf.position, world, self.rng)

    def _update_prey(
        self,
        animal: Animal,
        world: "WorldSim",
        agents: list["Agent"],
        wolves: list[Animal],
    ) -> None:
        # Build list of threat positions within flee_range.
        threats: list[tuple[int, int]] = []
        for agent in agents:
            if _manhattan(animal.position, agent.position) <= self.flee_range:
                threats.append(agent.position)
        for wolf in wolves:
            if wolf.alive and _manhattan(animal.position, wolf.position) <= self.flee_range:
                threats.append(wolf.position)

        if threats:
            nearest = min(threats, key=lambda t: _manhattan(animal.position, t))
            animal.position = _flee_from(animal.position, nearest, world, self.rng)
        else:
            animal.position = _random_move(animal.position, world, self.rng)
