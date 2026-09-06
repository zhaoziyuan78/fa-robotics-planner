import json
import sys
from pathlib import Path

import numpy as np
import torch

from fa_robotics_planner.baselines.env_adapter import (
    FlatObservationEnvAdapter,
    flatten_observation,
    resolve_task_setting,
    resolve_training_reward_mode,
)
from fa_robotics_planner.baselines.external import ExternalBaselineRunner
from fa_robotics_planner.baselines.offline_data import load_paired_data
from fa_robotics_planner.baselines.trajectory_transformer import (
    TrajectoryTransformer,
    _load_token_episodes,
    _top_factorized_actions,
    _truncate_after_first_success,
)
from fa_robotics_planner.data.writer import EpisodeWriter
from fa_robotics_planner.envs.windy import WindyControlEnv
from scripts.run_dino_wm_subprocess import _load_transitions


def test_active_baseline_configs_are_exactly_the_offline_protocol_set():
    root = Path(__file__).resolve().parents[1] / "configs" / "baseline"
    assert {path.stem for path in root.glob("*.yaml")} == {"gcrl", "tt", "dino_wm"}


def test_external_baseline_records_unsupported(tmp_path):
    status = ExternalBaselineRunner("dino_wm", None).run({}, tmp_path)
    assert status.status == "unsupported"
    assert (tmp_path / "baseline_status.json").exists()


def test_external_baseline_requires_summary(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import argparse,json,pathlib\n"
        "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\n"
        "o=pathlib.Path(a.output);(o/'summary.json').write_text(json.dumps({'status':'complete'}))\n"
    )
    status = ExternalBaselineRunner("dino_wm", [sys.executable, str(worker)]).run({}, tmp_path / "run")
    assert status.status == "complete"


def test_offline_loader_applies_an_exact_transition_budget(tmp_path):
    writer = EpisodeWriter(tmp_path / "paired", "paired")
    for episode_id in range(2):
        length = 2
        writer.write(
            episode_id,
            {
                "rgb": np.zeros((length, 4, 4, 3), np.uint8),
                "proprio": np.zeros((length, 1), np.float32),
                "control_state": np.full((length, 2), episode_id, np.float32),
                "state_mask": np.ones((length, 2), bool),
                "actions": np.zeros((length, 1), np.float32),
                "next_rgb": np.zeros((length, 4, 4, 3), np.uint8),
                "next_proprio": np.zeros((length, 1), np.float32),
                "next_control_state": np.full((length, 2), episode_id + 1, np.float32),
                "next_state_mask": np.ones((length, 2), bool),
                "rewards": np.zeros(length, np.float32),
                "terminated": np.zeros(length, bool),
                "truncated": np.zeros(length, bool),
                "goals": np.zeros((length, 1), np.float32),
                "sequence_length": np.asarray(length, np.int64),
            },
        )
    data = load_paired_data(tmp_path / "paired", 3)
    assert data.transition_count == 3
    assert [episode.length for episode in data.episodes] == [2, 1]


def _paired_episode(length: int, episode_id: int) -> dict[str, np.ndarray]:
    states = np.arange(length, dtype=np.float32)[:, None] + 10 * episode_id
    return {
        "rgb": np.full((length, 4, 4, 3), episode_id, np.uint8),
        "proprio": states.copy(),
        "control_state": states.copy(),
        "state_mask": np.ones((length, 1), bool),
        "actions": np.zeros((length, 1), np.float32),
        "next_rgb": np.full((length, 4, 4, 3), episode_id + 1, np.uint8),
        "next_proprio": states + 1,
        "next_control_state": states + 1,
        "next_state_mask": np.ones((length, 1), bool),
        "rewards": np.asarray([0, 1, 1, 1][:length], np.float32),
        "terminated": np.zeros(length, bool),
        "truncated": np.arange(length) == length - 1,
        "goals": np.zeros((length, 1), np.float32),
        "action_is_expert": np.ones(length, bool),
        "sequence_length": np.asarray(length, np.int64),
    }


def test_tt_terminal_view_removes_post_success_rewards_and_collector_label(tmp_path):
    writer = EpisodeWriter(tmp_path / "paired", "paired")
    writer.write(0, _paired_episode(4, 0))
    raw = load_paired_data(tmp_path / "paired", 4)
    cropped, removed = _truncate_after_first_success(raw, 1.0)
    assert cropped.episodes[0].length == 2
    assert cropped.episodes[0].dones.tolist() == [0.0, 1.0]
    assert cropped.episodes[0].action_is_expert is None
    assert removed == 2


def test_tt_pairs_vq_tokens_with_the_exact_partial_episode_view(tmp_path):
    paired = EpisodeWriter(tmp_path / "paired", "paired")
    paired.write(0, _paired_episode(4, 0))
    paired.write(1, _paired_episode(4, 1))
    tokens = EpisodeWriter(
        tmp_path / "tokens",
        "tokens",
        {
            "source": str((tmp_path / "paired").resolve()),
            "codebook_size": 8,
            "vqvae": str(tmp_path / "vq.pt"),
        },
    )
    for episode_id in range(2):
        values = np.arange(16, dtype=np.int64).reshape(4, 2, 2) % 8
        tokens.write(
            episode_id,
            {
                "video_tokens": values + 0,
                "next_video_tokens": (values + 1) % 8,
                "sequence_length": np.asarray(4, np.int64),
            },
        )
    data = load_paired_data(tmp_path / "paired", 6)
    episodes, metadata = _load_token_episodes(data, (tmp_path / "tokens").resolve())
    assert [episode.length for episode in episodes] == [4, 2]
    assert episodes[1].state_tokens.shape == (2, 4)
    assert metadata["token_shape"] == (2, 2)


def test_tt_model_predicts_discrete_actions_and_vq_transitions():
    model = TrajectoryTransformer(
        num_state_tokens=4,
        codebook_size=8,
        goal_dim=2,
        action_dim=3,
        action_bins=5,
        reward_bins=7,
        return_bins=9,
        d_model=16,
        state_embed_dim=4,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        context_length=5,
    )
    output = model(
        torch.randint(0, 8, (2, 3, 4)),
        torch.randn(2, 3, 2),
        torch.randint(0, 9, (2, 3)),
        torch.randint(0, 6, (2, 3, 3)),
        torch.randint(0, 8, (2, 3)),
        torch.randint(0, 3, (2, 3)),
        torch.randint(0, 5, (2, 3, 3)),
        torch.ones(2, 3, dtype=torch.bool),
    )
    assert output["action"].shape == (2, 3, 3, 5)
    assert output["next_state"].shape == (2, 3, 4, 8)
    assert output["reward"].shape == (2, 3, 7)
    assert output["done"].shape == (2, 3, 2)
    # Transition conditioning must retain actuator identity, not just the
    # unordered multiset of action bins.
    first = model._action_embedding(torch.tensor([[1, 2, 3]]))
    swapped = model._action_embedding(torch.tensor([[2, 1, 3]]))
    assert not torch.allclose(first, swapped)


def test_tt_joint_action_heap_matches_brute_force_order():
    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
    log_probabilities = torch.log_softmax(logits, dim=-1)
    actual = _top_factorized_actions(log_probabilities, candidates=5, top_k_per_dimension=3)
    brute_force = []
    for first in range(3):
        for second in range(3):
            score = float((log_probabilities[0, first] + log_probabilities[1, second]) / 2)
            brute_force.append((np.asarray([first, second]), score))
    brute_force.sort(key=lambda item: item[1], reverse=True)
    assert {tuple(tokens.tolist()) for tokens, _ in actual} == {
        tuple(tokens.tolist()) for tokens, _ in brute_force[:5]
    }
    assert np.allclose(
        [score for _, score in actual],
        sorted([score for _, score in brute_force], reverse=True)[:5],
    )


def test_dino_temporal_windows_never_cross_episode_boundaries(tmp_path):
    writer = EpisodeWriter(tmp_path / "paired", "paired")
    writer.write(0, _paired_episode(4, 0))
    writer.write(1, _paired_episode(4, 1))
    transitions = _load_transitions(tmp_path / "paired", 8, history_frames=3)
    assert transitions.window_starts.tolist() == [0, 1, 4, 5]
    assert transitions.history_frames == 3
    # desired_goal is a planning target, never a world-model input feature.
    assert transitions.current_proprio.shape[-1] == 1


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


def test_flat_baseline_dense_training_reward_preserves_native_reward():
    unified = WindyControlEnv({"episode_horizon": 2, "image_size": [32, 32]})
    env = FlatObservationEnvAdapter(
        unified,
        episode_horizon=2,
        initial_seed=7,
        reward_mode="dense_goal",
    )
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(2, np.float32))
    assert reward == -info["goal_distance"]
    assert info["native_reward"] == 0.0
    assert info["training_reward"] == reward
    assert resolve_training_reward_mode(
        {"training_reward_by_task": {"windy": "dense_goal"}}, "windy"
    ) == "dense_goal"
    assert resolve_training_reward_mode(
        {"training_reward_by_task": {"windy": "dense_goal"}}, "fetch_slide"
    ) == "native"


def test_baseline_task_setting_does_not_change_other_environments():
    config = {"train_ratio": 4.0, "train_ratio_by_task": {"windy": 64.0}}
    assert resolve_task_setting(config, "windy", "train_ratio", 1.0) == 64.0
    assert resolve_task_setting(config, "fetch_push", "train_ratio", 1.0) == 4.0


def test_flat_baseline_goal_progress_rewards_movement_toward_goal():
    unified = WindyControlEnv(
        {
            "episode_horizon": 10,
            "image_size": [32, 32],
            "w_max": 0.0,
            "gamma": 0.0,
        }
    )
    env = FlatObservationEnvAdapter(
        unified,
        episode_horizon=10,
        initial_seed=7,
        reward_mode="goal_progress",
        goal_progress_scale=50.0,
    )
    env.reset()
    _, toward_reward, _, _, toward_info = env.step(
        np.array([0.0, -1.0], np.float32)
    )
    assert toward_reward > 0.0
    assert toward_info["native_reward"] == 0.0
