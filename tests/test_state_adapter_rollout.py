import numpy as np
import torch

from fa_robotics_planner.config import compose_config
from fa_robotics_planner.models.builders import build_method
from scripts.eval_state_adapter_rollout import (
    _state_rendered_frames,
    metric_row,
    rollout_curves,
    rollout_from_history,
)


def _tiny_method():
    config = compose_config(
        [
            "model=prior_adapter",
            "env=windy",
            "env.image_size=[8,8]",
            "model.tokenizer.hidden_dim=16",
            "model.tokenizer.codebook_size=16",
            "model.tokenizer.code_dim=8",
            "model.state_prior.video.d_model=16",
            "model.state_prior.video.n_layers=1",
            "model.state_prior.video.n_heads=2",
            "model.state_prior.video.d_ff=32",
            "model.state_prior.observation.d_model=16",
            "model.state_prior.observation.n_layers=1",
            "model.state_prior.observation.n_heads=2",
            "model.state_adapter.hidden_dim=16",
        ]
    )
    return config, build_method(config).eval()


def _episode(length=4):
    states = np.arange(length * 4, dtype=np.float32).reshape(length, 4) / 20
    next_states = states + 0.05
    return {
        "rgb": np.zeros((length, 8, 8, 3), np.uint8),
        "control_state": states,
        "state_mask": np.ones_like(states, bool),
        "actions": np.zeros((length, 2), np.float32),
        "next_rgb": np.zeros((length, 8, 8, 3), np.uint8),
        "next_control_state": next_states,
        "next_state_mask": np.ones_like(states, bool),
        "sequence_length": np.asarray(length, np.int64),
        "action_is_expert": np.ones(length, bool),
    }


def test_open_loop_rollout_compares_passive_and_zero_initialized_adapter():
    config, method = _tiny_method()
    token_episode = {
        "video_tokens": np.zeros((4, 1, 1), np.int64),
        "next_video_tokens": np.zeros((4, 1, 1), np.int64),
    }
    result = rollout_from_history(
        method,
        _episode(),
        token_episode,
        history_steps=1,
        max_rollout_steps=2,
        device=torch.device("cpu"),
    )
    assert result["true_states"].shape == (3, 4)
    assert result["passive_tokens"].shape == (3, 1, 1)
    np.testing.assert_allclose(result["passive_states"], result["adapted_states"])
    np.testing.assert_array_equal(result["passive_tokens"], result["adapted_tokens"])
    curves = rollout_curves(result, config)
    row = metric_row(result, curves, episode_id=7)
    assert row["episode_id"] == 7
    assert row["rollout_steps"] == 2
    assert np.isfinite(row["adapted_state_rmse"])
    assert row["state_rmse_improvement_fraction"] == 0.0
    assert row["state_rmse_improvement_fraction_h1"] == 0.0
    assert row["state_rmse_improvement_fraction_h2"] == 0.0


def test_expert_only_rollout_stops_before_first_nonexpert_action():
    _, method = _tiny_method()
    episode = _episode()
    episode["action_is_expert"] = np.asarray([True, True, False, True])
    result = rollout_from_history(
        method,
        episode,
        None,
        history_steps=1,
        max_rollout_steps=0,
        device=torch.device("cpu"),
        expert_only=True,
    )
    assert result["rollout_steps"] == 1


def test_zero_rollout_limit_uses_complete_remaining_episode():
    _, method = _tiny_method()
    result = rollout_from_history(
        method,
        _episode(length=6),
        None,
        history_steps=2,
        max_rollout_steps=0,
        device=torch.device("cpu"),
    )
    assert result["rollout_steps"] == 4
    assert result["true_states"].shape[0] == 5


def test_windy_state_rendering_draws_the_moving_predicted_agent():
    config, _ = _tiny_method()
    states = np.asarray(
        [[-0.7, 0.7, 0.1, 0.0], [0.0, 0.0, 0.1, 0.0], [0.7, -0.7, 0.1, 0.0]],
        np.float32,
    )
    result = {
        "passive_states": states,
        "goal": np.asarray([0.8, -0.8], np.float32),
    }
    frames = _state_rendered_frames(result, config, "passive")
    assert len(frames) == 3
    assert frames[0].shape == (8, 8, 3)
    assert not np.array_equal(frames[0], frames[-1])


def test_robotics_state_rendering_animates_task_space_trajectory():
    config, _ = _tiny_method()
    config["env"] = {
        "name": "fetch_slide",
        "achieved_goal_slice": [2, 4],
    }
    states = np.asarray(
        [[0.0, 0.0, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
        np.float32,
    )
    result = {
        "adapted_states": states,
        "observed_states": states[:1],
        "true_states": states,
        "true_frames": np.zeros((3, 32, 32, 3), np.uint8),
        "goal": np.asarray([0.5, 0.5], np.float32),
    }
    frames = _state_rendered_frames(result, config, "adapted")
    assert len(frames) == 3
    assert not np.array_equal(frames[0], frames[-1])
