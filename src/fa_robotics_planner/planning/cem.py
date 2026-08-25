"""Continuous-action cross-entropy method using the common rollout/scorer."""

from __future__ import annotations

import torch

from .rollout import rollout_candidates
from .types import DynamicsModel, PlanningResult, TrajectoryScorer


class CEMPlanner:
    def __init__(
        self,
        horizon: int = 2,
        num_candidates: int = 256,
        num_elites: int = 32,
        iterations: int = 4,
        momentum: float = 0.1,
        min_std: float = 0.05,
        execute_steps: int = 1,
    ):
        if not 1 <= num_elites <= num_candidates:
            raise ValueError("num_elites must lie in [1, num_candidates]")
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.momentum = float(momentum)
        self.min_std = float(min_std)
        self.execute_steps = min(int(execute_steps), self.horizon)

    def plan(
        self,
        initial_state: torch.Tensor,
        goal: torch.Tensor,
        dynamics: DynamicsModel,
        scorer: TrajectoryScorer,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        initial_mean: torch.Tensor | None = None,
        initial_std: torch.Tensor | None = None,
        candidate_batch_size: int | None = None,
    ) -> PlanningResult:
        action_low, action_high = action_low.to(initial_state), action_high.to(initial_state)
        action_dim = action_low.numel()
        mean = initial_mean if initial_mean is not None else (action_low + action_high) * 0.5
        mean = torch.broadcast_to(mean, (self.horizon, action_dim)).clone()
        std = initial_std if initial_std is not None else (action_high - action_low) * 0.5
        std = torch.broadcast_to(std, (self.horizon, action_dim)).clone()
        total_calls = 0
        candidates = predicted = scores = None
        for _ in range(self.iterations):
            noise = torch.randn(self.num_candidates, self.horizon, action_dim, device=initial_state.device)
            candidates = torch.clamp(mean[None] + std[None] * noise, action_low, action_high)
            predicted, calls = rollout_candidates(
                dynamics,
                initial_state,
                candidates,
                candidate_batch_size=candidate_batch_size,
            )
            total_calls += calls
            scores = scorer(predicted, goal, candidates)
            elite_indices = torch.topk(scores, self.num_elites).indices
            elites = candidates[elite_indices]
            new_mean, new_std = elites.mean(0), elites.std(0, unbiased=False).clamp_min(self.min_std)
            mean = self.momentum * mean + (1 - self.momentum) * new_mean
            std = self.momentum * std + (1 - self.momentum) * new_std
        assert candidates is not None and predicted is not None and scores is not None
        best = int(torch.argmax(scores).item())
        return PlanningResult(candidates[best], best, scores, candidates, predicted, total_calls)

    def actions_to_execute(self, result: PlanningResult) -> torch.Tensor:
        return result.action_sequence[: self.execute_steps]
