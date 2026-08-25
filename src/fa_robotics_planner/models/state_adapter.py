"""Residual intervention model around frozen passive State Prior dynamics."""

from __future__ import annotations

import torch
from torch import nn


class StateAdapter(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        prior_hidden_dim: int,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = bool(residual)
        input_dim = state_dim * 2 + action_dim + prior_hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        current_state: torch.Tensor,
        passive_next: torch.Tensor,
        current_action: torch.Tensor,
        prior_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat((current_state, passive_next, current_action, prior_hidden), -1)
        delta = self.network(fused)
        predicted = passive_next + delta if self.residual else delta
        return predicted, delta
