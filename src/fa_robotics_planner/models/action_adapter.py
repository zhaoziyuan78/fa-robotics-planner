"""Residual correction of Action Prior distribution parameters."""

from __future__ import annotations

import torch
from torch import nn

from .distributions import TanhNormal


class ActionAdapter(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        goal_dim: int,
        prior_hidden_dim: int,
        hidden_dim: int,
        achieved_goal_slice: tuple[int, int] | None = None,
        normalize_goal_direction: bool = False,
        residual_scale: float = 0.5,
        residual_clip: float | None = 4.0,
        goal_feature_scale: float = 1.0,
    ):
        super().__init__()
        input_dim = 2 * action_dim + state_dim + goal_dim + prior_hidden_dim
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.achieved_goal_slice = achieved_goal_slice
        self.normalize_goal_direction = bool(normalize_goal_direction)
        self.residual_scale = float(residual_scale)
        self.residual_clip = (
            None if residual_clip is None else float(residual_clip)
        )
        self.goal_feature_scale = float(goal_feature_scale)
        if self.residual_scale <= 0:
            raise ValueError("Action Adapter residual_scale must be positive")
        if self.residual_clip is not None and self.residual_clip <= 0:
            raise ValueError("Action Adapter residual_clip must be positive or null")
        if self.goal_feature_scale <= 0:
            raise ValueError("Action Adapter goal_feature_scale must be positive")
        if achieved_goal_slice is not None:
            start, stop = achieved_goal_slice
            if not 0 <= start < stop <= state_dim or stop - start != goal_dim:
                raise ValueError(
                    "Action Adapter achieved_goal_slice must match goal_dim"
                )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        base_distribution: TanhNormal,
        prior_hidden: torch.Tensor,
        current_state: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[TanhNormal, torch.Tensor, torch.Tensor]:
        goal_features = goal
        if self.achieved_goal_slice is not None:
            start, stop = self.achieved_goal_slice
            goal_features = goal - current_state[..., start:stop]
            if self.normalize_goal_direction:
                goal_features = goal_features / torch.linalg.vector_norm(
                    goal_features, dim=-1, keepdim=True
                ).clamp_min(1e-6)
        goal_features = self.goal_feature_scale * goal_features
        features = torch.cat(
            (
                base_distribution.loc,
                base_distribution.log_scale,
                prior_hidden,
                current_state,
                goal_features,
            ),
            -1,
        )
        delta_mu, delta_log_scale = self.network(features).chunk(2, -1)
        if self.residual_clip is not None:
            delta_mu = delta_mu.clamp(-self.residual_clip, self.residual_clip)
            delta_log_scale = delta_log_scale.clamp(
                -self.residual_clip, self.residual_clip
            )
        delta_mu = self.residual_scale * delta_mu
        delta_log_scale = self.residual_scale * delta_log_scale
        corrected = TanhNormal(
            base_distribution.loc + delta_mu,
            (base_distribution.log_scale + delta_log_scale).clamp(-5.0, 2.0),
            base_distribution.low,
            base_distribution.high,
        )
        return corrected, delta_mu, delta_log_scale
