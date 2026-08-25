"""Prior-sampling/random-shooting receding-horizon planner."""

from __future__ import annotations

import torch

from .rollout import rollout_candidates
from .types import ActionProposal, DynamicsModel, PlanningResult, TrajectoryScorer


class RandomShootingPlanner:
    def __init__(
        self,
        horizon: int = 2,
        num_candidates: int = 256,
        execute_steps: int = 1,
        discount: float = 0.99,
        include_mean_candidate: bool = False,
    ):
        if horizon < 1 or num_candidates < 1 or execute_steps < 1:
            raise ValueError("horizon, num_candidates and execute_steps must be positive")
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.execute_steps = min(int(execute_steps), self.horizon)
        self.discount = float(discount)
        self.include_mean_candidate = bool(include_mean_candidate)

    def sample_candidates(
        self,
        initial_state: torch.Tensor,
        goal: torch.Tensor,
        proposal: ActionProposal,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        state = initial_state.reshape(1, -1).expand(self.num_candidates, -1)
        goal_batch = goal.reshape(1, -1).expand(self.num_candidates, -1)
        history = action_history
        if history.ndim == 2:
            history = history.unsqueeze(0).expand(self.num_candidates, -1, -1).clone()
        elif history.ndim == 3 and history.size(0) == 1:
            history = history.expand(self.num_candidates, -1, -1).clone()
        elif history.ndim != 3 or history.size(0) != self.num_candidates:
            raise ValueError("action_history must be [T,A], [1,T,A], or [K,T,A]")
        actions = []
        for _ in range(self.horizon):
            distribution = proposal(state, history, goal_batch)
            action = distribution.sample()
            if self.include_mean_candidate:
                action[0] = distribution.mean[0]
            actions.append(action)
            history = torch.cat((history, action.unsqueeze(1)), dim=1)
        return torch.stack(actions, dim=1)

    def plan(
        self,
        initial_state: torch.Tensor,
        goal: torch.Tensor,
        dynamics: DynamicsModel,
        scorer: TrajectoryScorer,
        *,
        proposal: ActionProposal | None = None,
        action_history: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> PlanningResult:
        if candidates is None:
            if proposal is None:
                raise ValueError("proposal is required when candidates are not supplied")
            if action_history is None:
                raise ValueError("action_history is required for prior sampling")
            candidates = self.sample_candidates(initial_state, goal, proposal, action_history)
        if candidates.shape[1] != self.horizon:
            raise ValueError(f"Expected candidate horizon {self.horizon}, got {candidates.shape[1]}")
        predicted, calls = rollout_candidates(dynamics, initial_state, candidates)
        return self.rank_candidates(candidates, predicted, goal, scorer, calls)

    def rank_candidates(
        self,
        candidates: torch.Tensor,
        predicted: torch.Tensor,
        goal: torch.Tensor,
        scorer: TrajectoryScorer,
        model_forward_calls: int = 0,
    ) -> PlanningResult:
        """Rank candidates whose state trajectories are already available.

        State-conditioned proposal generation may need to roll the dynamics
        forward before it can sample the next action.  Reusing that rollout
        here avoids evaluating exactly the same candidate sequence twice.
        """
        if candidates.ndim != 3:
            raise ValueError("candidates must have shape [K,H,A]")
        if predicted.ndim != 3:
            raise ValueError("predicted must have shape [K,H,S]")
        if candidates.shape[:2] != predicted.shape[:2]:
            raise ValueError("candidates and predicted trajectories must share [K,H]")
        if candidates.shape[1] != self.horizon:
            raise ValueError(f"Expected candidate horizon {self.horizon}, got {candidates.shape[1]}")
        scores = scorer(predicted, goal, candidates)
        if scores.shape != (candidates.shape[0],):
            raise ValueError("scorer must return one scalar per candidate")
        best = int(torch.argmax(scores).item())
        return PlanningResult(
            candidates[best],
            best,
            scores,
            candidates,
            predicted,
            int(model_forward_calls),
        )

    def actions_to_execute(self, result: PlanningResult) -> torch.Tensor:
        return result.action_sequence[: self.execute_steps]
