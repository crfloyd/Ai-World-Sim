"""Gymnasium-compatible RL environment wrapping WorldSim.

One environment instance = one world + one agent.
The environment handles:
  - world generation from a seeded pool (new seed per episode reset)
  - action dispatch to WorldSim
  - observation construction
  - action masking (included in the obs dict for RLlib)
  - episode termination (agent death or step limit)

Observation space (Dict):
  "action_mask"  Box float32 (NUM_ACTIONS,)  — 1=valid, 0=blocked
  "obs"          Box float32 (OBS_DIM,)      — flat observation vector

Action space:
  Discrete(6)
    0 move_north
    1 move_south
    2 move_east
    3 move_west
    4 forage
    5 rest

TODO: Extend to multi-agent (one env, N agents, shared policy via RLlib MARL).
TODO: Add rendering mode for visualisation / story replay.
TODO: Accept a list of seeds to cycle through deterministically in curriculum.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np

from ai_world_sim.common.config import load_config, DEFAULT_WORLD_CONFIG_PATH
from ai_world_sim.rl.observations import (
    NUM_ACTIONS,
    build_action_mask,
    build_observation,
    obs_dim,
)
from ai_world_sim.rl.rewards import survival_reward
from ai_world_sim.story.events import EventLog
from ai_world_sim.world.sim import WorldSim

# Action constants for readability
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
FORAGE = 4
REST = 5

_MOVE_DELTAS = {
    MOVE_NORTH: (-1, 0),
    MOVE_SOUTH: (1, 0),
    MOVE_EAST: (0, 1),
    MOVE_WEST: (0, -1),
}


class WorldEnv(gym.Env):
    """Single-agent survival environment backed by WorldSim."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict | None = None) -> None:
        super().__init__()
        cfg = env_config or {}

        # Load world config (can be overridden via env_config["world_config"]).
        world_config_path = cfg.get("world_config_path", DEFAULT_WORLD_CONFIG_PATH)
        self.world_config = load_config(world_config_path)
        if "world_config_override" in cfg:
            from ai_world_sim.common.config import merge_configs
            self.world_config = merge_configs(self.world_config, cfg["world_config_override"])

        self.window: int = int(cfg.get("observation_window", 5))
        self.max_steps: int = int(cfg.get("max_steps_per_episode", 500))
        self.seed_range: tuple[int, int] = tuple(cfg.get("seed_range", [0, 999_999]))

        self._obs_dim = obs_dim(self.window)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(NUM_ACTIONS,), dtype=np.float32
                ),
                "obs": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
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

        # --- Apply action ----------------------------------------------- #
        foraged = False
        action = int(action)

        if action in _MOVE_DELTAS:
            dr, dc = _MOVE_DELTAS[action]
            self._world.move_agent(self._agent, dr, dc)
        elif action == FORAGE:
            foraged = self._world.forage(self._agent)
        elif action == REST:
            self._world.rest(self._agent)

        # --- Advance world one tick ------------------------------------- #
        was_alive = self._agent.alive
        self._world.tick()
        died_this_step = was_alive and not self._agent.alive
        self._steps += 1

        # --- Reward ----------------------------------------------------- #
        reward = survival_reward(self._agent, died_this_step, foraged)

        # --- Termination ------------------------------------------------ #
        terminated = not self._agent.alive
        truncated = self._steps >= self.max_steps

        info: dict[str, Any] = {
            "day": self._world.day,
            "season": self._world.season.label(),
            "hunger": self._agent.hunger,
            "fatigue": self._agent.fatigue,
            "hp": self._agent.hp,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        pass  # TODO: ASCII or pygame renderer

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs_vec = build_observation(self._world, self._agent, window=self.window)
        mask = build_action_mask(self._world, self._agent)
        return {"action_mask": mask, "obs": obs_vec}

    @property
    def event_log(self) -> EventLog | None:
        return self._event_log
