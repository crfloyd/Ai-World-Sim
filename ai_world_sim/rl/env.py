"""Gymnasium-compatible RL environment wrapping WorldSim.

Observation space (Dict):
  "local_grid"      Box float32 (NUM_CHANNELS, WINDOW, WINDOW)
  "self_features"   Box float32 (SELF_DIM,)
  "memory_features" Box float32 (MEMORY_DIM,)
  "action_mask"     Box float32 (NUM_ACTIONS,)   1=valid, 0=blocked

Action space: Discrete(12)
  0 move_north   1 move_south   2 move_east    3 move_west
  4 forage       5 hunt         6 drink        7 eat
  8 rest         9 sleep       10 store_food  11 retrieve_food

Episode termination:
  terminated: agent died
  truncated:  max_steps_per_episode reached

TODO: Multi-agent variant (N agents, shared policy, one env).
TODO: Curriculum seeder (easy worlds early, harder worlds over training).
TODO: ASCII / pygame render mode for story visualization.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np

from ai_world_sim.common.config import DEFAULT_WORLD_CONFIG_PATH, load_config, merge_configs
from ai_world_sim.rl.observations import (
    MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST,
    FORAGE, HUNT, DRINK, EAT, REST, SLEEP, STORE_FOOD, RETRIEVE_FOOD,
    NUM_ACTIONS, NUM_CHANNELS, SELF_DIM,
    _MOVE_DELTAS, build_observation,
)
from ai_world_sim.rl.rewards import survival_reward
from ai_world_sim.story.events import EventLog
from ai_world_sim.world.memory import MEMORY_DIM
from ai_world_sim.world.sim import WorldSim


class WorldEnv(gym.Env):
    """Single-agent survival environment backed by WorldSim."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict | None = None) -> None:
        super().__init__()
        cfg = env_config or {}

        # Load world config.
        world_config_path = cfg.get("world_config_path", DEFAULT_WORLD_CONFIG_PATH)
        self.world_config = load_config(world_config_path)
        if "world_config_override" in cfg:
            self.world_config = merge_configs(
                self.world_config, cfg["world_config_override"]
            )

        mem_cfg = self.world_config.get("memory", {})
        self.window: int = int(2 * mem_cfg.get("sight_radius", 10) + 1)  # 21
        self.max_steps: int = int(cfg.get("max_steps_per_episode", 1000))
        self.seed_range: tuple[int, int] = tuple(cfg.get("seed_range", [0, 899_999]))

        # Gymnasium spaces.
        self.observation_space = gym.spaces.Dict(
            {
                "local_grid": gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(NUM_CHANNELS, self.window, self.window),
                    dtype=np.float32,
                ),
                "self_features": gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(SELF_DIM,),
                    dtype=np.float32,
                ),
                "memory_features": gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(MEMORY_DIM,),
                    dtype=np.float32,
                ),
                "action_mask": gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(NUM_ACTIONS,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        self._world: WorldSim | None = None
        self._agent = None
        self._steps = 0
        self._event_log: EventLog | None = None

    # ------------------------------------------------------------------ #
    # Gymnasium interface
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        world_seed = seed if seed is not None else random.randint(*self.seed_range)

        self._world = WorldSim(config=self.world_config, seed=world_seed)
        self._world.generate()

        self._event_log = EventLog()
        self._world.attach_event_log(self._event_log)

        agents = self._world.spawn_agents(n=1)
        self._agent = agents[0]
        self._steps = 0

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self._world is not None, "Call reset() before step()."
        assert self._agent is not None

        # Clear sleeping flag at start of step (any non-sleep action wakes the agent).
        if int(action) != SLEEP:
            self._agent.sleeping = False

        stored_food_this_step = False
        action = int(action)

        # --- Dispatch action -------------------------------------------- #
        if action in _MOVE_DELTAS:
            dr, dc = _MOVE_DELTAS[action]
            self._world.move_agent(self._agent, dr, dc)
        elif action == FORAGE:
            self._world.forage(self._agent)
        elif action == HUNT:
            self._world.hunt(self._agent)
        elif action == DRINK:
            self._world.drink(self._agent)
        elif action == EAT:
            self._world.eat(self._agent)
        elif action == REST:
            self._world.rest(self._agent)
        elif action == SLEEP:
            self._world.sleep(self._agent)
        elif action == STORE_FOOD:
            stored_food_this_step = self._world.store_food(self._agent)
        elif action == RETRIEVE_FOOD:
            self._world.retrieve_food(self._agent)

        # --- Advance world one tick ------------------------------------- #
        was_alive = self._agent.alive
        self._world.tick()
        died_this_step = was_alive and not self._agent.alive
        self._steps += 1

        # --- Reward ----------------------------------------------------- #
        reward = survival_reward(self._agent, died_this_step, stored_food_this_step, self.world_config)

        # --- Termination ------------------------------------------------ #
        terminated = not self._agent.alive
        truncated = self._steps >= self.max_steps

        info: dict[str, Any] = {
            "day": self._world.day,
            "season": self._world.season.label(),
            "hp": self._agent.hp,
            "hunger": self._agent.hunger,
            "thirst": self._agent.thirst,
            "tired": self._agent.tired,
            "steps": self._steps,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        pass  # TODO: ASCII renderer for story / debug mode

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> dict[str, np.ndarray]:
        return build_observation(self._world, self._agent, window=self.window)

    @property
    def event_log(self) -> EventLog | None:
        return self._event_log
