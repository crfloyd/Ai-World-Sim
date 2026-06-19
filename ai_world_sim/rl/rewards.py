"""Reward functions for the RL agent.

v0.1 uses a simple survival signal:
  +0.01 per tick alive        (encourages not dying)
  −0.001 × hunger             (discourages letting hunger rise)
  −0.001 × fatigue            (discourages exhaustion)
  −1.0   on death             (large terminal penalty)
  +0.05  per item foraged     (small shaping signal to encourage resource-seeking)

These weights are intentionally simple so they do not prescribe *how* to
survive — the agent must discover that foraging, resting, and movement
are instrumental to the survival objective.

TODO: Remove the forage shaping signal once the policy is stable; it may
      inadvertently bias toward pure hoarding over e.g. migration.
TODO: Add social reward signals once multi-agent systems exist.
TODO: Make reward weights config-driven.
"""

from __future__ import annotations

from ai_world_sim.world.entities import Agent

ALIVE_REWARD = 0.01
HUNGER_PENALTY_SCALE = 0.001
FATIGUE_PENALTY_SCALE = 0.001
DEATH_PENALTY = -1.0
FORAGE_BONUS = 0.05


def survival_reward(
    agent: Agent,
    died_this_step: bool,
    foraged_this_step: bool,
) -> float:
    """Compute one-step reward for *agent*.

    Parameters
    ----------
    agent:
        The agent after the step has been applied.
    died_this_step:
        True if the agent died during this step.
    foraged_this_step:
        True if the forage action succeeded.
    """
    if died_this_step:
        return DEATH_PENALTY

    reward = ALIVE_REWARD
    reward -= agent.hunger * HUNGER_PENALTY_SCALE
    reward -= agent.fatigue * FATIGUE_PENALTY_SCALE
    if foraged_this_step:
        reward += FORAGE_BONUS
    return float(reward)
