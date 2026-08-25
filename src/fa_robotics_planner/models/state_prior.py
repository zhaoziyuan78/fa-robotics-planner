"""Action-free causal multimodal State Prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StatePriorOutput:
    passive_next: torch.Tensor
    hidden: torch.Tensor
    visual_next: torch.Tensor
    proprio_next: torch.Tensor


class CausalStatePrior(nn.Module):
    """p(s_{t+1}|s_<=t) with visual/proprio/structured state tokens only."""

    def __init__(
        self,
        state_dim: int,
        proprio_dim: int = 0,
        visual_dim: int = 0,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 128,
        normalize_visual: bool = False,
        residual_prediction: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.proprio_dim = int(proprio_dim)
        self.visual_dim = int(visual_dim)
        self.max_length = int(max_length)
        self.normalize_visual = bool(normalize_visual)
        self.residual_prediction = bool(residual_prediction)
        self.state_embed = nn.Linear(2 * state_dim, d_model)
        self.proprio_embed = nn.Linear(proprio_dim, d_model) if proprio_dim else None
        self.visual_embed = nn.Linear(visual_dim, d_model) if visual_dim else None
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.state_head = nn.Linear(d_model, state_dim)
        self.proprio_head = nn.Linear(d_model, proprio_dim) if proprio_dim else None
        self.visual_head = nn.Linear(d_model, visual_dim) if visual_dim else None
        if self.residual_prediction:
            # Robot observations are strongly persistent at the environment
            # control rate.  Predicting only the change preserves the passive
            # prior definition while avoiding a regression-to-the-mean error
            # on absolute object/root positions.
            for head in (self.state_head, self.proprio_head, self.visual_head):
                if head is not None:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)

    def forward(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        proprio_sequence: torch.Tensor | None = None,
        visual_sequence: torch.Tensor | None = None,
        valid_steps: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> StatePriorOutput:
        if state_sequence.ndim != 3 or state_sequence.size(-1) != self.state_dim:
            raise ValueError(f"state_sequence must be [B,T,{self.state_dim}]")
        if state_mask.shape != state_sequence.shape:
            raise ValueError("state_mask must match state_sequence")
        batch, length, _ = state_sequence.shape
        masked = state_sequence * state_mask.to(state_sequence.dtype)
        tokens = self.state_embed(torch.cat((masked, state_mask.to(state_sequence.dtype)), -1))
        if self.proprio_embed is not None:
            if proprio_sequence is None or proprio_sequence.shape[:2] != (batch, length):
                raise ValueError("Configured State Prior requires matching proprio_sequence")
            tokens = tokens + self.proprio_embed(proprio_sequence)
        if self.visual_embed is not None:
            if visual_sequence is None or visual_sequence.shape[:2] != (batch, length):
                raise ValueError("Configured State Prior requires matching visual_sequence")
            tokens = tokens + self.visual_embed(visual_sequence)
        positions = torch.arange(
            int(position_offset), int(position_offset) + length, device=tokens.device
        ).clamp_max(self.max_length - 1)
        tokens = tokens + self.position(positions)[None]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1)
        padding = ~valid_steps.bool() if valid_steps is not None else None
        hidden = self.norm(self.transformer(tokens, mask=causal, src_key_padding_mask=padding))
        passive = self.state_head(hidden)
        if self.residual_prediction:
            passive = masked + passive
        proprio = self.proprio_head(hidden) if self.proprio_head is not None else passive.new_empty((*passive.shape[:2], 0))
        if self.proprio_head is not None and self.residual_prediction:
            assert proprio_sequence is not None
            proprio = proprio_sequence + proprio
        visual = self.visual_head(hidden) if self.visual_head is not None else passive.new_empty((*passive.shape[:2], 0))
        if self.visual_head is not None and self.residual_prediction:
            assert visual_sequence is not None
            visual = visual_sequence + visual
        if self.visual_head is not None and self.normalize_visual:
            visual = F.normalize(visual, dim=-1, eps=1e-6)
        return StatePriorOutput(passive, hidden, visual, proprio)

    def predict_next(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(*args, **kwargs)
        valid_steps = kwargs.get("valid_steps")
        if valid_steps is None and len(args) >= 5:
            valid_steps = args[4]
        if valid_steps is None:
            index = torch.full((output.hidden.size(0),), output.hidden.size(1) - 1, device=output.hidden.device, dtype=torch.long)
        else:
            index = valid_steps.long().sum(1).clamp_min(1) - 1
        batch = torch.arange(output.hidden.size(0), device=output.hidden.device)
        return output.passive_next[batch, index], output.hidden[batch, index]
