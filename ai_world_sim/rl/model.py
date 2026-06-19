"""Shared neural policy — actor-critic with separate CNN and MLP encoders.

Architecture:

  local_grid (NUM_CHANNELS, 21, 21)
      │
      └─► CNN encoder ──► global avg pool ──► cnn_embed (cnn_out_ch,)

  self_features (SELF_DIM,)
      │
      └─► MLP encoder ──► self_embed (self_mlp_hidden[-1],)

  memory_features (MEMORY_DIM,)
      │
      └─► MLP encoder ──► memory_embed (memory_mlp_hidden[-1],)

  concat [cnn_embed, self_embed, memory_embed]
      │
      └─► Shared trunk (MLP) ──► trunk_out

  trunk_out ──► policy_head ──► logits (NUM_ACTIONS,) [+ action masking]
           └─► value_head  ──► scalar value

Keeping grid and scalar inputs separate lets the CNN learn spatial patterns
without fighting the MLP for parameter budget.

The RLlib-compatible wrapper (ActionMaskModel) lives in train.py so this
module can be imported and tested without RLlib installed.

TODO: Add LSTM trunk variant for recurrent memory (gated temporal context).
TODO: Attention over a sequence of memory entries for richer memory access.
TODO: Experiment with multi-head policy (intent head + action head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_world_sim.rl.observations import NUM_ACTIONS, NUM_CHANNELS, SELF_DIM
from ai_world_sim.world.memory import MEMORY_DIM


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Build a linear stack: Linear → Activation → Linear → ..."""
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
    cnn_channels:
        Output channel counts for each Conv2d layer.
        A 3×3 same-padding conv is used for each entry.
    use_global_avg_pool:
        If True, apply global average pooling after the CNN so the CNN
        output size is independent of the observation window.
    self_mlp_hidden:
        Hidden sizes for the self-feature MLP encoder.
    memory_mlp_hidden:
        Hidden sizes for the memory-feature MLP encoder.
    trunk_hidden:
        Hidden sizes for the shared trunk MLP.
    """

    def __init__(
        self,
        cnn_channels: list[int] = (32, 64, 64),
        use_global_avg_pool: bool = True,
        self_mlp_hidden: list[int] = (128, 64),
        memory_mlp_hidden: list[int] = (64, 32),
        trunk_hidden: list[int] = (256, 128),
    ) -> None:
        super().__init__()

        # --- CNN encoder ----------------------------------------------- #
        cnn_layers: list[nn.Module] = []
        in_ch = NUM_CHANNELS
        for out_ch in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        if use_global_avg_pool:
            cnn_layers.append(nn.AdaptiveAvgPool2d(1))  # → (B, C, 1, 1)
            cnn_layers.append(nn.Flatten())
            cnn_out_dim = in_ch
        else:
            cnn_layers.append(nn.Flatten())
            # Output size unknown without window; caller must set correctly.
            cnn_out_dim = in_ch  # placeholder — trunk will be wrong without pool
        self.cnn = nn.Sequential(*cnn_layers)

        # --- Self-feature encoder --------------------------------------- #
        self.self_encoder = _mlp([SELF_DIM, *self_mlp_hidden], nn.ReLU)
        self_embed_dim = self_mlp_hidden[-1]

        # --- Memory-feature encoder ------------------------------------- #
        self.memory_encoder = _mlp([MEMORY_DIM, *memory_mlp_hidden], nn.ReLU)
        memory_embed_dim = memory_mlp_hidden[-1]

        # --- Shared trunk ----------------------------------------------- #
        trunk_in = cnn_out_dim + self_embed_dim + memory_embed_dim
        self.trunk = _mlp([trunk_in, *trunk_hidden], nn.ReLU)
        trunk_out_dim = trunk_hidden[-1]

        # --- Heads ------------------------------------------------------ #
        self.policy_head = nn.Linear(trunk_out_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(trunk_out_dim, 1)

    def forward(
        self,
        local_grid: torch.Tensor,
        self_features: torch.Tensor,
        memory_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        local_grid:
            Shape (B, NUM_CHANNELS, W, W).
        self_features:
            Shape (B, SELF_DIM).
        memory_features:
            Shape (B, MEMORY_DIM).
        action_mask:
            Shape (B, NUM_ACTIONS), float32 — 1=valid, 0=blocked.

        Returns
        -------
        logits:
            Shape (B, NUM_ACTIONS).
        value:
            Shape (B,).
        """
        cnn_embed = self.cnn(local_grid)
        self_embed = F.relu(self.self_encoder(self_features))
        memory_embed = F.relu(self.memory_encoder(memory_features))

        combined = torch.cat([cnn_embed, self_embed, memory_embed], dim=-1)
        trunk_out = self.trunk(combined)

        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)

        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * -1e9

        return logits, value

    def act(
        self,
        local_grid: torch.Tensor,
        self_features: torch.Tensor,
        memory_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Returns (action, log_prob, value).
        """
        logits, value = self.forward(local_grid, self_features, memory_features, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value
