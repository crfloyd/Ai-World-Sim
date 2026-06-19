"""Shared neural policy — actor-critic architecture.

Architecture:
  Input: flat obs vector split into grid (first GRID_FLAT dims) and state (rest).

  CNN encoder:
    Reshape grid → (B, C, W, W) → Conv layers → flatten → cnn_embedding

  MLP encoder:
    state → Linear layers → state_embedding

  Trunk:
    concat(cnn_embedding, state_embedding) → shared Linear layers → trunk_out

  Policy head:
    trunk_out → Linear → raw_logits
    + action_mask applied (subtract large value from blocked actions)

  Value head:
    trunk_out → Linear(1) → scalar

This model is the standalone PyTorch version.  The RLlib-compatible wrapper
(which subclasses TorchModelV2) lives in train.py to keep this module
importable without RLlib as a dependency.

TODO: Add LSTM / GRU trunk variant for temporal memory.
TODO: Add attention over inventory tokens.
TODO: Experiment with multi-head for separate "intent" vs "action" outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_world_sim.rl.observations import NUM_ACTIONS, NUM_GRID_CHANNELS, STATE_DIM


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class AgentBrain(nn.Module):
    """Actor-critic model shared across all agent instances.

    Parameters
    ----------
    window:
        Side length of the local grid window (e.g. 5 → 5×5 grid).
    cnn_channels:
        List of (out_channels, kernel_size) for each Conv2d layer.
    mlp_hidden:
        Hidden layer sizes for the state MLP encoder.
    trunk_hidden:
        Hidden layer sizes for the shared trunk after concatenation.
    """

    def __init__(
        self,
        window: int = 5,
        cnn_channels: list[tuple[int, int]] = ((16, 3), (32, 3)),
        mlp_hidden: list[int] = (128, 64),
        trunk_hidden: list[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.window = window
        self.num_channels = NUM_GRID_CHANNELS

        # --- CNN encoder ----------------------------------------------- #
        cnn_layers: list[nn.Module] = []
        in_ch = NUM_GRID_CHANNELS
        for out_ch, k in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = in_ch * window * window

        # --- MLP encoder for agent state -------------------------------- #
        self.state_encoder = _mlp([STATE_DIM, *mlp_hidden], nn.ReLU)
        state_embed_dim = mlp_hidden[-1]

        # --- Shared trunk ----------------------------------------------- #
        trunk_in = cnn_out_dim + state_embed_dim
        self.trunk = _mlp([trunk_in, *trunk_hidden], nn.ReLU)
        trunk_out_dim = trunk_hidden[-1]

        # --- Heads ------------------------------------------------------ #
        self.policy_head = nn.Linear(trunk_out_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(trunk_out_dim, 1)

        self._grid_flat = NUM_GRID_CHANNELS * window * window

    def _split_obs(
        self, flat_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a flat observation into grid tensor and state vector."""
        grid_flat = flat_obs[:, : self._grid_flat]
        state = flat_obs[:, self._grid_flat :]
        # Reshape grid: (B, H*W*C) → (B, C, H, W)
        grid = grid_flat.view(-1, self.num_channels, self.window, self.window)
        return grid, state

    def forward(
        self,
        flat_obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        flat_obs:
            Shape (B, OBS_DIM) — the concatenated grid + state observation.
        action_mask:
            Shape (B, NUM_ACTIONS), float32 — 1=valid, 0=blocked.
            If provided, blocked actions are set to −1e9 before returning.

        Returns
        -------
        logits:
            Shape (B, NUM_ACTIONS) — masked policy logits.
        value:
            Shape (B,) — state value estimate.
        """
        grid, state = self._split_obs(flat_obs)

        cnn_embed = self.cnn(grid)
        state_embed = F.relu(self.state_encoder(state))

        trunk_in = torch.cat([cnn_embed, state_embed], dim=-1)
        trunk_out = self.trunk(trunk_in)

        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)

        if action_mask is not None:
            # Large negative value effectively zeros the probability of masked actions.
            logits = logits + (1.0 - action_mask) * -1e9

        return logits, value

    def act(
        self,
        flat_obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Returns (action, log_prob, value).
        """
        logits, value = self.forward(flat_obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
