import json
import sys

import numpy as np

from fa_robotics_planner.baselines.env_adapter import (
    FlatObservationEnvAdapter,
    flatten_observation,
)
from fa_robotics_planner.baselines.external import ExternalBaselineRunner
from fa_robotics_planner.baselines.gcrl import (
    GoalEnvAdapter,
    _resolved_learning_starts,
    _resolved_replay_buffer_size,
)
from fa_robotics_planner.envs.windy import WindyControlEnv


def test_external_baseline_records_unsupported(tmp_path):
    status = ExternalBaselineRunner("dreamerv3", None).run({}, tmp_path)
    assert status.status == "unsupported"
    assert (tmp_path / "baseline_status.json").exists()


def test_external_baseline_requires_summary(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import argparse,json,pathlib\n"
        "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\n"
        "o=pathlib.Path(a.output);(o/'summary.json').write_text(json.dumps({'status':'complete'}))\n"
    )
    status = ExternalBaselineRunner("tdmpc2", [sys.executable, str(worker)]).run({}, tmp_path / "run")
    assert status.status == "complete"


def test_flat_baseline_adapter_normalizes_actions_and_advances_seeds():
    unified = WindyControlEnv({"episode_horizon": 2, "image_size": [32, 32]})
    env = FlatObservationEnvAdapter(unified, episode_horizon=2, initial_seed=7)
    first, first_info = env.reset()
    second, second_info = env.reset()
    assert first.shape == env.observation_space.shape
    assert first_info["seed"] == 7
    assert second_info["seed"] == 8
    observation, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
    assert observation.shape == first.shape
    assert not terminated
    assert not truncated
    assert {"success", "success_subtasks", "is_terminal"} <= info.keys()


def test_flat_baseline_observation_does_not_duplicate_proprio_or_mask():
    unified = WindyControlEnv({"episode_horizon": 2, "image_size": [32, 32]})
    bundle = unified.reset(4)
    flattened = flatten_observation(bundle)
    assert flattened.shape == (bundle.control_state.size + bundle.goal.size,)
    assert np.array_equal(flattened[: bundle.control_state.size], bundle.control_state)


def test_gcrl_her_waits_for_a_complete_episode():
    assert _resolved_learning_starts(100, 50, use_her=True) == 100
    assert _resolved_learning_starts(100, 500, use_her=True) == 501
    assert _resolved_learning_starts(100, 500, use_her=False) == 100


def test_gcrl_her_buffer_cannot_wrap_during_a_short_budget_run():
    assert _resolved_replay_buffer_size(None, 1_000, 500, use_her=True) == 1_001
    assert _resolved_replay_buffer_size(100, 1_000, 500, use_her=True) == 1_001
    assert _resolved_replay_buffer_size(2_000, 1_000, 500, use_her=True) == 2_000
    assert _resolved_replay_buffer_size(100, 1_000, 500, use_her=False) == 100


def test_gcrl_real_and_relabelled_rewards_use_the_same_sparse_convention():
    unified = WindyControlEnv(
        {"episode_horizon": 2, "image_size": [32, 32], "success_radius": 0.1}
    )
    env = GoalEnvAdapter(unified, episode_horizon=2, use_goal_reward=True)
    observation, _ = env.reset(seed=7)
    next_observation, reward, _, _, info = env.step(np.zeros(2, np.float32))
    relabelled_reward = env.compute_reward(
        next_observation["achieved_goal"], observation["desired_goal"], info
    )
    assert reward == float(relabelled_reward) == -1.0
    assert info["native_reward"] == 0.0
