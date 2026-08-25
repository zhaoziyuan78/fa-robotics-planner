"""Uniform continuous and discrete action-distribution interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class ActionDistribution(Protocol):
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor: ...
    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...
    @property
    def mean(self) -> torch.Tensor: ...


@dataclass
class TanhNormal:
    loc: torch.Tensor
    log_scale: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor

    def __post_init__(self) -> None:
        self.log_scale = self.log_scale.clamp(-5.0, 2.0)
        self.low = torch.as_tensor(self.low, dtype=self.loc.dtype, device=self.loc.device)
        self.high = torch.as_tensor(self.high, dtype=self.loc.dtype, device=self.loc.device)
        if torch.any(self.high <= self.low):
            raise ValueError("Action high bounds must be greater than low bounds")

    @property
    def base(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.loc, self.log_scale.exp())

    def _to_action(self, raw: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(raw)
        return self.low + (unit + 1.0) * 0.5 * (self.high - self.low)

    @property
    def mean(self) -> torch.Tensor:
        return self._to_action(self.loc)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._to_action(self.base.rsample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        scale = 0.5 * (self.high - self.low)
        unit = ((value - self.low) / scale - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(unit)
        jacobian = torch.log(scale) + torch.log1p(-unit.square() + 1e-6)
        return (self.base.log_prob(raw) - jacobian).sum(dim=-1)


@dataclass
class CategoricalActions:
    logits: torch.Tensor
    action_table: torch.Tensor | None = None

    @property
    def categorical(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)

    @property
    def mean(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        if self.action_table is None:
            return torch.argmax(probabilities, dim=-1)
        table = self.action_table.to(self.logits)
        return probabilities @ table

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        indices = self.categorical.sample(sample_shape)
        if self.action_table is None:
            return indices
        return self.action_table.to(self.logits)[indices]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self.action_table is not None and value.shape[-1:] == self.action_table.shape[-1:]:
            table = self.action_table.to(value)
            value = torch.square(value.unsqueeze(-2) - table).sum(-1).argmin(-1)
        return self.categorical.log_prob(value.long())

    def probabilities(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

