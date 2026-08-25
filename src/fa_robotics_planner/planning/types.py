from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class DynamicsModel(Protocol):
    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


class TrajectoryScorer(Protocol):
    def __call__(self, states: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor: ...


class ActionProposal(Protocol):
    def __call__(self, state: torch.Tensor, history: torch.Tensor, goal: torch.Tensor): ...


@dataclass
class PlanningResult:
    action_sequence: torch.Tensor
    best_index: int
    scores: torch.Tensor
    candidates: torch.Tensor
    predicted_states: torch.Tensor
    model_forward_calls: int

    @property
    def first_action(self) -> torch.Tensor:
        return self.action_sequence[0]

