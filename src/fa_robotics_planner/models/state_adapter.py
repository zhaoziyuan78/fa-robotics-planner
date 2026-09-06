"""Action-conditioned corrections for observation and discrete video dynamics."""

from __future__ import annotations

import torch
from torch import nn


class StateAdapter(nn.Module):
    """Apply one shared action condition to both State-Prior branches."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        observation_hidden_dim: int,
        video_hidden_dim: int,
        codebook_size: int,
        tokens_per_frame: int,
        residual: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.observation_hidden_dim = int(observation_hidden_dim)
        self.video_hidden_dim = int(video_hidden_dim)
        self.codebook_size = int(codebook_size)
        self.tokens_per_frame = int(tokens_per_frame)
        self.residual = bool(residual)
        condition_dim = (
            2 * self.state_dim
            + self.action_dim
            + self.observation_hidden_dim
            + self.video_hidden_dim
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.state_head = nn.Linear(self.hidden_dim, self.state_dim)
        self.condition_to_video = nn.Linear(self.hidden_dim, self.video_hidden_dim)
        self.video_spatial = nn.Embedding(
            self.tokens_per_frame, self.video_hidden_dim
        )
        self.video_head = nn.Sequential(
            nn.LayerNorm(self.video_hidden_dim),
            nn.Linear(self.video_hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.codebook_size),
        )
        for module in (self.state_head, self.video_head[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_condition(
        self,
        current_state: torch.Tensor,
        passive_next: torch.Tensor,
        current_action: torch.Tensor,
        observation_hidden: torch.Tensor,
        video_summary: torch.Tensor,
    ) -> torch.Tensor:
        return self.condition(
            torch.cat(
                (
                    current_state,
                    passive_next,
                    current_action,
                    observation_hidden,
                    video_summary,
                ),
                dim=-1,
            )
        )

    def forward(
        self,
        current_state: torch.Tensor,
        passive_next: torch.Tensor,
        current_action: torch.Tensor,
        observation_hidden: torch.Tensor,
        video_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = self.encode_condition(
            current_state,
            passive_next,
            current_action,
            observation_hidden,
            video_summary,
        )
        delta = self.state_head(condition)
        predicted = passive_next + delta if self.residual else delta
        return predicted, delta, condition

    def adapt_video_logits(
        self,
        base_logits: torch.Tensor,
        token_hidden: torch.Tensor,
        spatial_index: int | torch.Tensor,
        condition: torch.Tensor,
        *,
        projected_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(spatial_index, int):
            spatial = self.video_spatial.weight[spatial_index]
        else:
            spatial = self.video_spatial(spatial_index.long())
        while spatial.ndim < token_hidden.ndim:
            spatial = spatial.unsqueeze(0)
        if projected_condition is None:
            projected_condition = self.condition_to_video(condition)
        conditioned_hidden = (
            token_hidden
            + projected_condition
            + spatial
        )
        correction = self.video_head(conditioned_hidden)
        return base_logits + correction, correction

    def adapt_video_sequence(
        self,
        base_logits: torch.Tensor,
        token_hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized teacher-forced correction for [...,N,V] tensors."""

        if base_logits.shape[:-1] != token_hidden.shape[:-1]:
            raise ValueError("base_logits and token_hidden leading shapes differ")
        if base_logits.size(-1) != self.codebook_size:
            raise ValueError("base_logits has the wrong codebook dimension")
        if token_hidden.size(-2) != self.tokens_per_frame:
            raise ValueError("token_hidden has the wrong frame-token dimension")
        spatial = self.video_spatial(
            torch.arange(self.tokens_per_frame, device=token_hidden.device)
        )
        leading = [1] * (token_hidden.ndim - 2)
        spatial = spatial.view(*leading, self.tokens_per_frame, self.video_hidden_dim)
        video_condition = self.condition_to_video(condition).unsqueeze(-2)
        correction = self.video_head(token_hidden + video_condition + spatial)
        return base_logits + correction, correction
