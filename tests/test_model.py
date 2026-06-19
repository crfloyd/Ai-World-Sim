"""Tests for AgentBrain model architecture and inference."""

from __future__ import annotations

import torch
import pytest

from ai_world_sim.rl.model import AgentBrain
from ai_world_sim.rl.observations import NUM_ACTIONS, NUM_CHANNELS, SELF_DIM
from ai_world_sim.world.memory import MEMORY_DIM


@pytest.fixture()
def brain():
    return AgentBrain()


def _dummy_inputs(batch=1, window=21):
    grid = torch.zeros(batch, NUM_CHANNELS, window, window)
    self_f = torch.zeros(batch, SELF_DIM)
    mem_f = torch.zeros(batch, MEMORY_DIM)
    mask = torch.ones(batch, NUM_ACTIONS)
    return grid, self_f, mem_f, mask


def test_forward_output_shapes(brain):
    grid, self_f, mem_f, mask = _dummy_inputs()
    logits, value = brain(grid, self_f, mem_f, mask)
    assert logits.shape == (1, NUM_ACTIONS)
    assert value.shape == (1,)


def test_action_mask_blocks_invalid(brain):
    grid, self_f, mem_f, _ = _dummy_inputs()
    mask = torch.zeros(1, NUM_ACTIONS)
    mask[0, 8] = 1.0  # only REST is valid
    logits, _ = brain(grid, self_f, mem_f, mask)
    # REST should have the highest logit by far.
    assert logits.argmax(dim=-1).item() == 8


def test_act_stochastic_returns_valid_action(brain):
    grid, self_f, mem_f, mask = _dummy_inputs()
    action, log_prob, value = brain.act(grid, self_f, mem_f, mask, deterministic=False)
    assert 0 <= action.item() < NUM_ACTIONS


def test_act_deterministic_uses_argmax(brain):
    """Deterministic act must use torch.argmax, not dist.mode."""
    grid, self_f, mem_f, _ = _dummy_inputs()
    # Force one action to be clearly best by biasing the mask.
    mask = torch.zeros(1, NUM_ACTIONS)
    mask[0, 3] = 1.0  # only MOVE_WEST is valid
    logits, _ = brain(grid, self_f, mem_f, mask)
    expected = torch.argmax(logits, dim=-1).item()

    action, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=True)
    assert action.item() == expected


def test_act_deterministic_consistent(brain):
    """Two deterministic calls with the same input must return the same action."""
    grid, self_f, mem_f, mask = _dummy_inputs()
    action1, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=True)
    action2, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=True)
    assert action1.item() == action2.item()


def test_batch_forward(brain):
    grid, self_f, mem_f, mask = _dummy_inputs(batch=4)
    logits, value = brain(grid, self_f, mem_f, mask)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4,)
