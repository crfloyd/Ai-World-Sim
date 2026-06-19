"""Reward functions for the RL agent.

V0 rewards are purely outcome-based (survival pressure).

Rules:
  - Never reward farming, hunting, foraging, or specific behaviors directly.
  - Only reward outcomes: staying alive, maintaining health, not starving,
    not dehydrating, storing food as a buffer against future scarcity.
  - The policy must discover HOW to achieve these outcomes itself.

All reward weights are read from ``config["rewards"]`` so they can be
tuned without touching code.  The module-level constants below are the
documented defaults — they are used as fallbacks when no config is supplied.

Current reward signal:
  +alive_reward           per tick alive                          (default 0.01)
  -hunger_penalty_scale × hunger                                  (default 0.001)
  -thirst_penalty_scale × thirst   (higher than hunger weight)   (default 0.0015)
  -tired_penalty_scale × tired                                    (default 0.0005)
  +store_food_bonus       on a successful store_food action       (default 0.05)
  +death_penalty          on death (negative value)               (default -1.0)

The ``store_food_bonus`` is a temporary shaping signal to encourage
planning ahead.  Set it to 0.0 in the config to disable it once the
policy is stable.

TODO: Remove store_food_bonus once policy reliably returns home to store.
TODO: Add living-through-winter bonus once seasonal difficulty is noticeable.
"""

from __future__ import annotations

from ai_world_sim.world.entities import Agent

# Default weights — used when no config is supplied.
ALIVE_REWARD = 0.01
HUNGER_PENALTY_SCALE = 0.001
THIRST_PENALTY_SCALE = 0.0015       # slightly higher: thirst kills faster than hunger
TIRED_PENALTY_SCALE = 0.0005
STORE_FOOD_BONUS = 0.05             # temporary shaping; set config rewards.store_food_bonus=0.0 to disable
DEATH_PENALTY = -1.0


def survival_reward(
    agent: Agent,
    died_this_step: bool,
    stored_food_this_step: bool,
    config: dict | None = None,
) -> float:
    """Compute one-step reward for *agent*.

    Parameters
    ----------
    agent:
        The agent after the step has been applied.
    died_this_step:
        True if the agent transitioned alive → dead this step.
    stored_food_this_step:
        True if the store_food action succeeded this step.
    config:
        World/training config dict.  Reads weights from ``config["rewards"]``.
        Falls back to module-level defaults when None or key is missing.
    """
    cfg = (config or {}).get("rewards", {})
    alive_reward = float(cfg.get("alive_reward", ALIVE_REWARD))
    hunger_scale = float(cfg.get("hunger_penalty_scale", HUNGER_PENALTY_SCALE))
    thirst_scale = float(cfg.get("thirst_penalty_scale", THIRST_PENALTY_SCALE))
    tired_scale = float(cfg.get("tired_penalty_scale", TIRED_PENALTY_SCALE))
    store_bonus = float(cfg.get("store_food_bonus", STORE_FOOD_BONUS))
    death_penalty = float(cfg.get("death_penalty", DEATH_PENALTY))

    if died_this_step:
        return death_penalty

    reward = alive_reward
    reward -= agent.hunger * hunger_scale
    reward -= agent.thirst * thirst_scale
    reward -= agent.tired * tired_scale
    if stored_food_this_step:
        reward += store_bonus
    return float(reward)
