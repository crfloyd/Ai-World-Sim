"""Reward functions for the RL agent.

V0 rewards are purely outcome-based (survival pressure).

Rules:
  - Never reward farming, hunting, foraging, or specific behaviors directly.
  - Only reward outcomes: staying alive, maintaining health, not starving,
    not dehydrating, storing food as a buffer against future scarcity.
  - The policy must discover HOW to achieve these outcomes itself.

Current reward signal:
  +0.01  per tick alive
  -0.001 × hunger      (encourages keeping hunger low)
  -0.001 × thirst      (encourages keeping thirst low; slightly higher weight)
  -0.0005 × tired      (mild discouragement of exhaustion)
  +0.1   on a day survived   (milestone bonus every ticks_per_day ticks)
  +0.05  on storing food     (shaping: rewards planning ahead)
  -1.0   on death

Weights are intentionally conservative — a surviving agent should earn
a small but clearly positive signal, and death should dominate as negative.

TODO: Remove the store_food shaping signal once the policy is stable.
TODO: Make all weights config-driven.
TODO: Add living-through-winter bonus once seasonal difficulty is noticeable.
"""

from __future__ import annotations

from ai_world_sim.world.entities import Agent

ALIVE_REWARD = 0.01
HUNGER_PENALTY_SCALE = 0.001
THIRST_PENALTY_SCALE = 0.0015       # slightly higher than hunger: thirst is faster-killing
TIRED_PENALTY_SCALE = 0.0005
STORE_FOOD_BONUS = 0.05
DEATH_PENALTY = -1.0


def survival_reward(
    agent: Agent,
    died_this_step: bool,
    stored_food_this_step: bool,
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
    """
    if died_this_step:
        return DEATH_PENALTY

    reward = ALIVE_REWARD
    reward -= agent.hunger * HUNGER_PENALTY_SCALE
    reward -= agent.thirst * THIRST_PENALTY_SCALE
    reward -= agent.tired * TIRED_PENALTY_SCALE
    if stored_food_this_step:
        reward += STORE_FOOD_BONUS
    return float(reward)
