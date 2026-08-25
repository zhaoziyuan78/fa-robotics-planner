from pathlib import Path

import numpy as np
import pytest
import torch

from fa_robotics_planner.config import compose_config
from fa_robotics_planner.planning import CEMPlanner, RandomShootingPlanner
from fa_robotics_planner.planning.rollout import rollout_candidates
from fa_robotics_planner.planning.scorers import (
    fetch_score,
    humanoid_score,
    humanoid_score_torch,
)


ROOT = Path(__file__).resolve().parents[1]


def dynamics(state, action):
    return state + action


def scorer(states, goal, actions):
    return -torch.square(states[:, -1] - goal).sum(-1)


def test_candidate_ranking_horizon_one_and_bounds():
    planner = RandomShootingPlanner(horizon=1, num_candidates=3)
    candidates = torch.tensor([[[-1.0]], [[0.5]], [[2.0]]])
    result = planner.plan(torch.tensor([0.0]), torch.tensor([0.5]), dynamics, scorer, candidates=candidates)
    assert result.best_index == 1
    assert torch.equal(result.first_action, torch.tensor([0.5]))
    assert planner.actions_to_execute(result).shape == (1, 1)


def test_horizon_two_and_receding_horizon_first_action_only():
    planner = RandomShootingPlanner(horizon=2, num_candidates=2, execute_steps=1)
    candidates = torch.tensor([[[0.2], [0.3]], [[0.4], [0.4]]])
    result = planner.plan(torch.tensor([0.0]), torch.tensor([0.5]), dynamics, scorer, candidates=candidates)
    assert result.best_index == 0
    assert torch.equal(planner.actions_to_execute(result), torch.tensor([[0.2]]))


def test_rank_candidates_reuses_precomputed_rollout():
    planner = RandomShootingPlanner(horizon=2, num_candidates=2)
    candidates = torch.tensor([[[0.2], [0.3]], [[0.4], [0.4]]])
    predicted, calls = rollout_candidates(dynamics, torch.tensor([0.0]), candidates)
    result = planner.rank_candidates(
        candidates, predicted, torch.tensor([0.5]), scorer, calls
    )
    assert result.best_index == 0
    assert result.model_forward_calls == 2
    assert torch.equal(result.predicted_states, predicted)


def test_random_shooting_can_anchor_the_proposal_mean():
    class FixedDistribution:
        def __init__(self, batch):
            self.mean = torch.full((batch, 1), 0.25)

        def sample(self):
            return torch.full_like(self.mean, -0.5)

    def proposal(state, history, goal):
        return FixedDistribution(state.size(0))

    planner = RandomShootingPlanner(
        horizon=2, num_candidates=3, include_mean_candidate=True
    )
    candidates = planner.sample_candidates(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        proposal,
        torch.empty(0, 1),
    )
    assert torch.equal(candidates[0], torch.tensor([[0.25], [0.25]]))
    assert torch.equal(candidates[1], torch.tensor([[-0.5], [-0.5]]))


def test_terminal_branch_stops_rollout():
    def terminal_dynamics(state, action):
        next_state = state + action
        return next_state, next_state[:, 0] >= 1.0

    candidates = torch.tensor([[[1.0], [10.0]], [[0.2], [0.2]]])
    states, _ = rollout_candidates(terminal_dynamics, torch.tensor([0.0]), candidates)
    assert states[0, -1, 0] == 1.0
    assert states[1, -1, 0] == 0.4


def test_cem_respects_action_bounds():
    torch.manual_seed(0)
    planner = CEMPlanner(horizon=2, num_candidates=64, num_elites=8, iterations=2)
    result = planner.plan(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        dynamics,
        scorer,
        torch.tensor([-0.25]),
        torch.tensor([0.25]),
        candidate_batch_size=17,
    )
    assert torch.all(result.candidates <= 0.25)
    assert torch.all(result.candidates >= -0.25)
    assert result.model_forward_calls == 16


def test_batched_rollout_matches_individual_rollout():
    candidates = torch.tensor([[[0.1], [0.2]], [[-0.3], [0.5]], [[0.0], [0.7]]])
    batched, _ = rollout_candidates(dynamics, torch.tensor([0.2]), candidates)
    individual = []
    for candidate in candidates:
        state = torch.tensor([0.2])
        trajectory = []
        for action in candidate:
            state = dynamics(state, action)
            trajectory.append(state)
        individual.append(torch.stack(trajectory))
    assert torch.allclose(batched, torch.stack(individual))


def test_candidate_rollout_batching_preserves_order_and_bounds_peak_batch():
    candidates = torch.tensor(
        [[[0.1], [0.2]], [[-0.3], [0.5]], [[0.0], [0.7]], [[0.4], [-0.1]]]
    )
    observed_batch_sizes = []

    def recording_dynamics(state, action):
        observed_batch_sizes.append(state.size(0))
        return state + action

    expected, expected_calls = rollout_candidates(
        dynamics, torch.tensor([0.2]), candidates
    )
    actual, calls = rollout_candidates(
        recording_dynamics,
        torch.tensor([0.2]),
        candidates,
        candidate_batch_size=3,
    )
    assert torch.equal(actual, expected)
    assert calls == expected_calls * 2
    assert max(observed_batch_sizes) == 3


@pytest.mark.parametrize(
    "task", ("humanoid_stand", "humanoid_balance", "humanoid_reach", "humanoid_push")
)
def test_vectorized_humanoid_score_matches_scalar_score(task):
    torch.manual_seed(2)
    env_config = compose_config([f"env={task}"], config_root=ROOT / "configs")["env"]
    states = torch.randn(7, 512)
    actions = torch.randn(7, 2, 61)
    goal_size = int(env_config.get("goal_size", 0))
    goal = torch.randn(goal_size)
    expected = np.asarray(
        [
            humanoid_score(
                task,
                states[index].numpy(),
                goal.numpy(),
                actions[index].numpy(),
                env_config["scorer"],
            )
            for index in range(states.size(0))
        ]
    )
    actual = humanoid_score_torch(task, states, goal, actions, env_config["scorer"])
    assert np.allclose(actual.numpy(), expected, atol=1e-5)


def test_fetch_score_reads_achieved_goal_before_padding():
    states = torch.zeros(2, 1, 64)
    states[0, 0, 25:28] = torch.tensor([1.0, 2.0, 3.0])
    states[1, 0, 25:28] = torch.tensor([4.0, 5.0, 6.0])
    # Deliberately make the padded tail misleading: the old implementation
    # read this location and ranked the two candidates identically.
    states[:, 0, -3:] = torch.tensor([9.0, 9.0, 9.0])
    goal = torch.tensor([1.0, 2.0, 3.0])
    actions = torch.zeros(2, 1, 4)
    scores = fetch_score(
        "fetch_push",
        states,
        goal,
        actions,
        {
            "achieved_goal_slice": [25, 28],
            "gripper_position_slice": [0, 3],
            "approach_cost": 0.0,
        },
    )
    assert scores[0] > scores[1]
    assert scores[0] == 0
