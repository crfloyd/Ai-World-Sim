"""Tests for config-driven reward weights and survival_reward behavior."""

from __future__ import annotations

from ai_world_sim.rl.rewards import (
    survival_reward,
    ALIVE_REWARD,
    HUNGER_PENALTY_SCALE,
    THIRST_PENALTY_SCALE,
    TIRED_PENALTY_SCALE,
    STORE_FOOD_BONUS,
    DEATH_PENALTY,
)
from ai_world_sim.world.entities import Agent


def _agent(**kwargs) -> Agent:
    defaults = dict(id=0, position=(0, 0), home_position=(0, 0), hp=100.0,
                    hunger=0.0, thirst=0.0, tired=0.0)
    defaults.update(kwargs)
    return Agent(**defaults)


def test_death_returns_death_penalty():
    a = _agent()
    r = survival_reward(a, died_this_step=True, stored_food_this_step=False)
    assert r == DEATH_PENALTY


def test_alive_with_no_stress_returns_alive_reward():
    a = _agent(hunger=0.0, thirst=0.0, tired=0.0)
    r = survival_reward(a, died_this_step=False, stored_food_this_step=False)
    assert abs(r - ALIVE_REWARD) < 1e-9


def test_store_food_bonus_applied():
    a = _agent()
    r = survival_reward(a, died_this_step=False, stored_food_this_step=True)
    assert r == ALIVE_REWARD + STORE_FOOD_BONUS


def test_config_overrides_alive_reward():
    a = _agent()
    cfg = {"rewards": {"alive_reward": 0.5}}
    r = survival_reward(a, died_this_step=False, stored_food_this_step=False, config=cfg)
    assert abs(r - 0.5) < 1e-9


def test_config_overrides_death_penalty():
    a = _agent()
    cfg = {"rewards": {"death_penalty": -5.0}}
    r = survival_reward(a, died_this_step=True, stored_food_this_step=False, config=cfg)
    assert r == -5.0


def test_config_overrides_store_food_bonus():
    a = _agent()
    cfg = {"rewards": {"store_food_bonus": 0.0}}
    r_no_bonus = survival_reward(a, died_this_step=False, stored_food_this_step=True, config=cfg)
    r_default = survival_reward(a, died_this_step=False, stored_food_this_step=True)
    assert r_no_bonus < r_default


def test_hunger_penalty_scales_with_stat():
    a_hungry = _agent(hunger=100.0)
    a_fed = _agent(hunger=0.0)
    r_hungry = survival_reward(a_hungry, False, False)
    r_fed = survival_reward(a_fed, False, False)
    assert r_hungry < r_fed


def test_thirst_penalty_higher_than_hunger():
    """Thirst penalty scale > hunger penalty scale at equal stat values."""
    a = _agent(hunger=50.0, thirst=0.0, tired=0.0)
    b = _agent(hunger=0.0, thirst=50.0, tired=0.0)
    r_hungry = survival_reward(a, False, False)
    r_thirsty = survival_reward(b, False, False)
    # Thirst should incur a larger penalty.
    assert r_thirsty < r_hungry


def test_no_config_uses_defaults():
    a = _agent(hunger=0.0, thirst=0.0, tired=0.0)
    r = survival_reward(a, False, False, config=None)
    assert r == ALIVE_REWARD
