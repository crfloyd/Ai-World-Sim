"""WorldSim — the central simulation object.

WorldSim owns the grid, all agents, and all sub-systems.  It is intentionally
decoupled from the RL environment so it can be used for:
  - standalone observation/story runs
  - headless batch simulations
  - future multi-agent or client-server architectures

Tick vs. Day:
  A *tick* is the finest time unit (one agent action slot).
  A *day* is ``ticks_per_day`` ticks.  Season advances every ``days_per_season``
  days.  The RL env calls ``step()`` once per tick.

TODO: Add support for multiple simultaneous agents per world.
TODO: Add trade routes and inter-agent interaction systems.
TODO: Add crime / conflict system (theft, assault, reputation).
TODO: Add faction / settlement system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ai_world_sim.world.entities import Agent
from ai_world_sim.world.generator import find_spawn_positions, generate_world
from ai_world_sim.world.systems.ecology import EcologySystem
from ai_world_sim.world.systems.health import HealthSystem
from ai_world_sim.world.systems.resources import ResourceSystem
from ai_world_sim.world.systems.seasons import Season, SeasonSystem
from ai_world_sim.world.terrain import Cell

if TYPE_CHECKING:
    from ai_world_sim.story.events import EventLog


class WorldSim:
    """Container for one complete procedurally generated world instance."""

    def __init__(self, config: dict, seed: int | None = None) -> None:
        self.config = config
        world_cfg = config.get("world", {})

        self.width: int = int(world_cfg.get("width", 64))
        self.height: int = int(world_cfg.get("height", 64))
        self.ticks_per_day: int = int(world_cfg.get("ticks_per_day", 24))
        self.seed: int = seed if seed is not None else int(world_cfg.get("default_seed", 42))

        self._rng = np.random.default_rng(self.seed)

        self.tick_count: int = 0
        self.day: int = 0
        self._tick_within_day: int = 0

        # Systems
        self._season_sys = SeasonSystem(
            days_per_season=int(world_cfg.get("days_per_season", 30))
        )
        self._health_sys = HealthSystem(config)
        self._resource_sys = ResourceSystem(config, self._rng)
        self._ecology_sys = EcologySystem(config)

        self.season: Season = Season.SPRING
        self.grid: list[list[Cell]] = []
        self.agents: list[Agent] = []

        self.event_log: EventLog | None = None

    # ------------------------------------------------------------------ #
    # World generation
    # ------------------------------------------------------------------ #

    def generate(self, seed: int | None = None) -> None:
        """(Re-)generate the world from *seed* (defaults to self.seed)."""
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

        self.tick_count = 0
        self.day = 0
        self._tick_within_day = 0
        self.season = Season.SPRING
        self.agents = []

        self.grid = generate_world(
            width=self.width,
            height=self.height,
            seed=self.seed,
            config=self.config,
        )

    def attach_event_log(self, log: "EventLog") -> None:
        self.event_log = log

    # ------------------------------------------------------------------ #
    # Agent management
    # ------------------------------------------------------------------ #

    def spawn_agents(self, n: int = 1) -> list[Agent]:
        """Spawn *n* agents on random passable cells and return them."""
        positions = find_spawn_positions(self.grid, n, self._rng)
        agent_cfg = self.config.get("agents", {})
        new_agents = []
        start_id = len(self.agents)
        for i, pos in enumerate(positions):
            agent = Agent(
                id=start_id + i,
                position=pos,
                hp=float(agent_cfg.get("start_hp", 100.0)),
                hunger=float(agent_cfg.get("start_hunger", 0.0)),
                fatigue=float(agent_cfg.get("start_fatigue", 0.0)),
                traits={
                    "strength": float(self._rng.uniform(0.4, 1.0)),
                    "agility": float(self._rng.uniform(0.4, 1.0)),
                    "perception": float(self._rng.uniform(0.4, 1.0)),
                },
                skills={"foraging": 0.0},
                # TODO: Add more initial traits and skills as systems expand.
            )
            new_agents.append(agent)
        self.agents.extend(new_agents)
        return new_agents

    # ------------------------------------------------------------------ #
    # Time advancement
    # ------------------------------------------------------------------ #

    def tick(self) -> dict[str, Any]:
        """Advance the simulation by one tick.

        Returns a dict of events that occurred this tick (season change, etc.)
        so callers can react without having to diff state.
        """
        events: dict[str, Any] = {}

        # 1. Health/hunger/fatigue decay for all living agents.
        for agent in self.agents:
            self._health_sys.tick(agent, self.event_log)

        # 2. Resource regeneration.
        self._resource_sys.tick(self.grid, self.season)

        # 3. Ecology (placeholder for v0.1).
        self._ecology_sys.tick(self)

        # 4. Advance time.
        self.tick_count += 1
        self._tick_within_day += 1

        if self._tick_within_day >= self.ticks_per_day:
            self._tick_within_day = 0
            old_day = self.day
            self.day += 1
            if self._season_sys.did_season_change(old_day, self.day):
                self.season = self._season_sys.season_for_day(self.day)
                events["season_change"] = self.season
                if self.event_log is not None:
                    self.event_log.log(f"Season changed to {self.season.label()}.")

        return events

    def advance_day(self) -> list[dict[str, Any]]:
        """Run a full day's worth of ticks. Returns the list of per-tick event dicts."""
        tick_events = []
        for _ in range(self.ticks_per_day):
            tick_events.append(self.tick())
        return tick_events

    # ------------------------------------------------------------------ #
    # Cell accessors
    # ------------------------------------------------------------------ #

    def cell(self, row: int, col: int) -> Cell:
        return self.grid[row][col]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_passable(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row][col].is_passable()

    # ------------------------------------------------------------------ #
    # Agent actions (called by the RL environment)
    # ------------------------------------------------------------------ #

    def move_agent(self, agent: Agent, dr: int, dc: int) -> bool:
        """Attempt to move *agent* by (dr, dc). Returns True on success."""
        nr, nc = agent.position[0] + dr, agent.position[1] + dc
        if not self.is_passable(nr, nc):
            return False
        agent.position = (nr, nc)
        if self.event_log:
            direction = {(-1, 0): "north", (1, 0): "south",
                         (0, 1): "east", (0, -1): "west"}.get((dr, dc), "?")
            self.event_log.log(f"Agent {agent.id} moved {direction}.")
        return True

    def forage(self, agent: Agent) -> bool:
        """Agent attempts to gather resources from their current cell.

        Returns True if anything was gathered.
        """
        r, c = agent.position
        cell = self.grid[r][c]
        agent_cfg = self.config.get("agents", {})
        gathered = False

        if cell.berries > 0:
            amount = min(cell.berries, int(agent_cfg.get("forage_berry_amount", 2)))
            cell.berries -= amount
            agent.add_item("berries", amount)
            self._health_sys.apply_eat(agent, amount)
            gathered = True
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} foraged {amount} berries.")

        elif cell.trees > 0:
            amount = min(cell.trees, int(agent_cfg.get("forage_wood_amount", 1)))
            cell.trees -= amount
            agent.add_item("wood", amount)
            gathered = True
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} gathered {amount} wood.")

        elif cell.stone > 0:
            amount = min(cell.stone, int(agent_cfg.get("forage_stone_amount", 1)))
            cell.stone -= amount
            agent.add_item("stone", amount)
            gathered = True
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} gathered {amount} stone.")

        return gathered

    def rest(self, agent: Agent) -> None:
        """Agent rests, recovering fatigue."""
        agent_cfg = self.config.get("agents", {})
        recovery = float(agent_cfg.get("rest_fatigue_recovery", 5.0))
        self._health_sys.apply_rest(agent, recovery)
        if self.event_log:
            self.event_log.log(f"Agent {agent.id} rested.")
