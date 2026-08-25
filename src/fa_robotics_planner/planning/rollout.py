"""Shared batched rollout that freezes terminal branches."""

from __future__ import annotations

import torch

from .types import DynamicsModel


def rollout_candidates(
    dynamics: DynamicsModel,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
    candidate_batch_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    if action_sequences.ndim != 3:
        raise ValueError("action_sequences must have shape [K,H,A]")
    candidates = action_sequences.size(0)
    if candidate_batch_size is None or candidate_batch_size >= candidates:
        return _rollout_candidate_batch(dynamics, initial_state, action_sequences)
    candidate_batch_size = int(candidate_batch_size)
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    predicted_batches = []
    calls = 0
    for start in range(0, candidates, candidate_batch_size):
        predicted, batch_calls = _rollout_candidate_batch(
            dynamics,
            initial_state,
            action_sequences[start : start + candidate_batch_size],
        )
        predicted_batches.append(predicted)
        calls += batch_calls
    return torch.cat(predicted_batches, dim=0), calls


def _rollout_candidate_batch(
    dynamics: DynamicsModel,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Roll out a single batch; split orchestration stays outside the loop."""

    candidates, horizon, _ = action_sequences.shape
    state = initial_state.reshape(1, -1).expand(candidates, -1).clone()
    terminal = torch.zeros(candidates, dtype=torch.bool, device=state.device)
    states = []
    calls = 0
    for step in range(horizon):
        prediction = dynamics(state, action_sequences[:, step])
        calls += 1
        if isinstance(prediction, tuple):
            next_state, became_terminal = prediction
            became_terminal = became_terminal.bool().reshape(-1)
        else:
            next_state = prediction
            became_terminal = torch.zeros_like(terminal)
        state = torch.where(terminal[:, None], state, next_state)
        terminal = terminal | became_terminal
        states.append(state)
    return torch.stack(states, dim=1), calls
