"""WorldSim — the central simulation object.

WorldSim owns the grid, all agents, all animals, and all sub-systems.
It is intentionally decoupled from the RL environment so it can be used for:
  - standalone observation / story runs
  - headless batch simulations
  - future multi-agent or client-server architectures

Tick vs. Day:
  A *tick* is the finest time unit (one agent action slot).
  A *day* is ``ticks_per_day`` ticks.  Season advances every
  ``days_per_season`` days.  The RL env calls ``step()`` once per tick.

Action dispatch is intentionally on WorldSim so non-RL code (story mode,
unit tests) can drive agents without the Gymnasium wrapper.

TODO: Add support for multiple simultaneous neural agents per world.
TODO: Add trade system (inter-agent item exchange).
TODO: Add crime / conflict system (theft, assault, territorial behavior).
TODO: Add faction / settlement system.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_world_sim.common.config import resolve_predator_profile
from ai_world_sim.world.animals import Animal, AnimalSpecies
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.generator import find_spawn_positions, generate_world
from ai_world_sim.world.systems.animals import AnimalSystem
from ai_world_sim.world.systems.ecology import EcologySystem
from ai_world_sim.world.systems.health import HealthSystem
from ai_world_sim.world.systems.resources import ResourceSystem
from ai_world_sim.world.systems.seasons import Season, SeasonSystem
from ai_world_sim.world.terrain import Cell, TerrainType


class WorldSim:
    """Container for one complete procedurally generated world instance."""

    def __init__(self, config: dict, seed: int | None = None) -> None:
        self.config = resolve_predator_profile(config)
        world_cfg = self.config.get("world", {})
        mem_cfg = self.config.get("memory", {})

        self.width: int = int(world_cfg.get("width", 64))
        self.height: int = int(world_cfg.get("height", 64))
        self.ticks_per_day: int = int(world_cfg.get("ticks_per_day", 96))
        self.seed: int = seed if seed is not None else int(world_cfg.get("default_seed", 42))
        self.sight_radius: int = int(mem_cfg.get("sight_radius", 10))

        self._rng = np.random.default_rng(self.seed)

        self.tick_count: int = 0
        self.day: int = 0
        self._tick_within_day: int = 0

        # Sub-systems
        self._season_sys = SeasonSystem(
            days_per_season=int(world_cfg.get("days_per_season", 30))
        )
        self._health_sys = HealthSystem(self.config)
        self._resource_sys = ResourceSystem(self.config, self._rng)
        self._ecology_sys = EcologySystem(self.config)
        self._animal_sys = AnimalSystem(self.config, self._rng)

        self.season: Season = Season.SPRING
        self.grid: list[list[Cell]] = []
        self.agents: list[Agent] = []
        self.animals: list[Animal] = []

        self.event_log = None  # attached by caller; avoids circular import

        # Diagonal of the world for normalisation (used by memory summariser).
        self.world_diag: float = float((self.height ** 2 + self.width ** 2) ** 0.5)

    # ------------------------------------------------------------------ #
    # World generation
    # ------------------------------------------------------------------ #

    def generate(self, seed: int | None = None) -> None:
        """(Re-)generate the world and spawn scripted animals."""
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

        self.tick_count = 0
        self.day = 0
        self._tick_within_day = 0
        self.season = Season.SPRING
        self.agents = []
        self.animals = []

        self.grid = generate_world(
            width=self.width,
            height=self.height,
            seed=self.seed,
            config=self.config,
        )
        self._spawn_animals()

    def _spawn_animals(self) -> None:
        """Spawn scripted animals at random passable positions."""
        anim_cfg = self.config.get("animals", {})
        counts = {
            AnimalSpecies.RABBIT: int(anim_cfg.get("rabbits_per_world", 20)),
            AnimalSpecies.DEER: int(anim_cfg.get("deer_per_world", 10)),
            AnimalSpecies.WOLF: int(anim_cfg.get("wolves_per_world", 3)),
        }
        total = sum(counts.values())
        try:
            positions = find_spawn_positions(self.grid, total, self._rng)
        except RuntimeError:
            positions = []

        idx = 0
        animal_id = 0
        for species, count in counts.items():
            for _ in range(count):
                if idx >= len(positions):
                    break
                self.animals.append(
                    Animal.create(animal_id, positions[idx], species, self.config)
                )
                animal_id += 1
                idx += 1

    def attach_event_log(self, log: Any) -> None:
        self.event_log = log

    # ------------------------------------------------------------------ #
    # Agent management
    # ------------------------------------------------------------------ #

    def spawn_agents(self, n: int = 1) -> list[Agent]:
        """Spawn *n* agents on random passable cells and return them."""
        positions = find_spawn_positions(self.grid, n, self._rng)
        agent_cfg = self.config.get("agents", {})
        mem_cfg = self.config.get("memory", {})
        from ai_world_sim.world.memory import MemoryStore

        new_agents: list[Agent] = []
        start_id = len(self.agents)
        for i, pos in enumerate(positions):
            mem_store = MemoryStore(
                max_entries=int(mem_cfg.get("max_entries", 20)),
                decay_rate=float(mem_cfg.get("decay_rate", 0.02)),
            )
            agent = Agent(
                id=start_id + i,
                position=pos,
                home_position=pos,
                hp=float(agent_cfg.get("start_hp", 100.0)),
                hunger=float(agent_cfg.get("start_hunger", 0.0)),
                thirst=float(agent_cfg.get("start_thirst", 0.0)),
                tired=float(agent_cfg.get("start_tired", 0.0)),
                traits={
                    "strength": float(self._rng.uniform(0.4, 1.0)),
                    "agility": float(self._rng.uniform(0.4, 1.0)),
                    "perception": float(self._rng.uniform(0.4, 1.0)),
                },
                skills={"foraging": 0.0, "hunting": 0.0},
                memory=mem_store,
                # TODO: sample traits from a distribution that creates
                #       meaningful individual variation.
            )
            new_agents.append(agent)
        self.agents.extend(new_agents)
        for agent in new_agents:
            self.update_agent_memory(agent)
        return new_agents

    def update_agent_memory(self, agent: Agent) -> None:
        """Scan the agent's visible area and refresh memory, then decay stale entries."""
        agent.memory.update(
            grid=self.grid,
            agent_pos=agent.position,
            animals=self.animals,
            tick=self.tick_count,
            sight_radius=self.sight_radius,
            world_height=self.height,
            world_width=self.width,
        )
        agent.memory.decay(self.tick_count)

    # ------------------------------------------------------------------ #
    # Time advancement
    # ------------------------------------------------------------------ #

    def tick(self) -> dict[str, Any]:
        """Advance simulation by one tick. Returns events dict."""
        events: dict[str, Any] = {}

        # 1. Health decay for all living agents.
        for agent in self.agents:
            self._health_sys.tick(agent, self.event_log)

        # 2. Scripted animal behavior.
        self._animal_sys.tick(self, self.animals, self.agents, self.event_log)

        # 3. Resource regeneration.
        self._resource_sys.tick(self.grid, self.season)

        # 4. Ecological processes (placeholder v0.1).
        self._ecology_sys.tick(self)

        # 5. Advance clock.
        self.tick_count += 1
        self._tick_within_day += 1

        if self._tick_within_day >= self.ticks_per_day:
            self._tick_within_day = 0
            old_day = self.day
            self.day += 1
            if self._season_sys.did_season_change(old_day, self.day):
                self.season = self._season_sys.season_for_day(self.day)
                events["season_change"] = self.season
                if self.event_log:
                    self.event_log.log(f"Season changed to {self.season.label()}.")

        # 6. Perception: each living agent updates memory from current world state.
        for agent in self.agents:
            if agent.alive:
                self.update_agent_memory(agent)

        return events

    def advance_day(self) -> list[dict[str, Any]]:
        """Run a full day of ticks. Returns per-tick event dicts."""
        return [self.tick() for _ in range(self.ticks_per_day)]

    # ------------------------------------------------------------------ #
    # Grid accessors
    # ------------------------------------------------------------------ #

    def cell(self, row: int, col: int) -> Cell:
        return self.grid[row][col]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_passable(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row][col].is_passable()

    def adjacent_water(self, row: int, col: int) -> bool:
        """True if any cardinal neighbour is a water tile."""
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc].is_water():
                return True
        return False

    def adjacent_prey(self, row: int, col: int) -> Animal | None:
        """Return the first adjacent living prey animal, or None."""
        for animal in self.animals:
            if not animal.alive or not animal.is_prey:
                continue
            if abs(animal.position[0] - row) + abs(animal.position[1] - col) == 1:
                return animal
        return None

    # ------------------------------------------------------------------ #
    # Agent actions (called by the RL environment or story mode)
    # ------------------------------------------------------------------ #

    def move_agent(self, agent: Agent, dr: int, dc: int) -> bool:
        """Attempt to move *agent* by (dr, dc). Returns True on success."""
        nr, nc = agent.position[0] + dr, agent.position[1] + dc
        if not self.is_passable(nr, nc):
            return False
        agent.position = (nr, nc)
        agent.sleeping = False
        # Moving costs extra tiredness.
        tired_cost = float(self.config.get("agents", {}).get("tired_per_move", 0.15))
        self._health_sys.apply_tired_move_cost(agent, tired_cost)
        if self.event_log:
            direction = {(-1, 0): "north", (1, 0): "south",
                         (0, 1): "east", (0, -1): "west"}.get((dr, dc), "?")
            self.event_log.log(f"Agent {agent.id} moved {direction}.")
        return True

    def forage(self, agent: Agent) -> bool:
        """Gather resources from the current cell into inventory."""
        r, c = agent.position
        cell = self.grid[r][c]
        agent_cfg = self.config.get("agents", {})
        gathered = False

        if cell.berries > 0:
            amount = min(cell.berries, int(agent_cfg.get("forage_berry_amount", 2)))
            cell.berries -= amount
            agent.add_item("berries", amount)
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

        agent.sleeping = False
        return gathered

    def eat(self, agent: Agent) -> bool:
        """Consume one food item from inventory to reduce hunger."""
        agent.sleeping = False
        if agent.remove_item("berries", 1):
            self._health_sys.apply_eat_berries(agent)
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} ate berries.")
            return True
        if agent.remove_item("meat", 1):
            self._health_sys.apply_eat_meat(agent)
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} ate meat.")
            return True
        return False

    def drink(self, agent: Agent) -> bool:
        """Drink from an adjacent water tile to reduce thirst."""
        r, c = agent.position
        agent.sleeping = False
        if self.adjacent_water(r, c):
            self._health_sys.apply_drink(agent)
            if self.event_log:
                self.event_log.log(f"Agent {agent.id} drank water.")
            return True
        return False

    def hunt(self, agent: Agent) -> bool:
        """Hunt an adjacent prey animal. Adds meat to inventory on kill."""
        agent.sleeping = False
        r, c = agent.position
        prey = self.adjacent_prey(r, c)
        if prey is None:
            return False
        prey.alive = False
        agent.add_item("meat", prey.meat_yield)
        if self.event_log:
            self.event_log.log(
                f"Agent {agent.id} hunted {prey.species.value} and gained {prey.meat_yield} meat."
            )
        return True

    def rest(self, agent: Agent) -> None:
        """Short-term recovery: small tired and HP restoration."""
        agent.sleeping = False
        self._health_sys.apply_rest(agent)
        if self.event_log:
            self.event_log.log(f"Agent {agent.id} rested.")

    def sleep(self, agent: Agent) -> None:
        """Long-term recovery: large tired + HP restoration; sets sleeping flag.

        While sleeping=True, wolves deal bonus damage during this tick's
        animal update (which runs after actions in env.step).
        """
        agent.sleeping = True
        self._health_sys.apply_sleep(agent)
        location = "at home" if agent.is_at_home else "outside"
        if self.event_log:
            self.event_log.log(f"Agent {agent.id} slept {location}.")

    def store_food(self, agent: Agent) -> bool:
        """Move all food from agent's inventory to home storage.

        Only valid when the agent is at home_position.
        """
        agent.sleeping = False
        if not agent.is_at_home:
            return False
        stored = agent.store_all_food()
        if stored > 0 and self.event_log:
            self.event_log.log(f"Agent {agent.id} stored {stored} food at home.")
        return stored > 0

    def retrieve_food(self, agent: Agent) -> bool:
        """Retrieve up to *retrieve_food_amount* units of stored food.

        Only valid when the agent is at home_position.
        """
        agent.sleeping = False
        if not agent.is_at_home:
            return False
        agent_cfg = self.config.get("agents", {})
        amount = int(agent_cfg.get("retrieve_food_amount", 1))
        retrieved = agent.retrieve_up_to(amount)
        if retrieved > 0 and self.event_log:
            self.event_log.log(f"Agent {agent.id} retrieved {retrieved} stored food.")
        return retrieved > 0
