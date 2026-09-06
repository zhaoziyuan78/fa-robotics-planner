import json
import sys

import numpy as np
import pytest
import torch

from fa_robotics_planner.data import DatasetKind, EpisodeWriter, LazyEpisodeDataset, check_dataset
from fa_robotics_planner.data.generate import generate_episode
from fa_robotics_planner.data.schemas import validate_episode
from fa_robotics_planner.envs.unified import ObservationBundle, StepResult
from fa_robotics_planner.models.vqvae import VQVAE
from scripts import train_prior
from scripts.train_prior import _EarlyStopper


def action_episode(length=4):
    return {
        "actions": np.zeros((length, 2), np.float32),
        "sequence_length": np.asarray(length, np.int64),
        "action_low": np.full(2, -1, np.float32),
        "action_high": np.full(2, 1, np.float32),
    }


def state_episode(episode_id, length=4):
    rng = np.random.default_rng(episode_id)
    control_state = rng.normal(0, 0.1, size=(length, 4)).astype(np.float32)
    return {
        "rgb": np.zeros((length, 8, 8, 3), np.uint8),
        "proprio": control_state[:, 2:].copy(),
        "control_state": control_state,
        "state_mask": np.ones((length, 4), bool),
        "sequence_length": np.asarray(length, np.int64),
    }


def test_action_only_rejects_information_leakage():
    episode = action_episode()
    episode["goal"] = np.zeros(2)
    with pytest.raises(ValueError, match="forbidden"):
        validate_episode(DatasetKind.ACTION_ONLY, episode)


def test_lazy_dataset_manifest_and_integrity(tmp_path):
    writer = EpisodeWriter(tmp_path, DatasetKind.ACTION_ONLY)
    writer.write(0, action_episode())
    writer.write(1, action_episode(3))
    assert check_dataset(tmp_path) == []
    dataset = LazyEpisodeDataset(tmp_path)
    assert len(dataset) == 2
    assert dataset[1]["actions"].shape == (3, 2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["kind"] == "action_prior"


def test_lazy_dataset_limits_transitions_by_whole_episode(tmp_path):
    writer = EpisodeWriter(tmp_path, DatasetKind.ACTION_ONLY)
    for episode_id in range(5):
        writer.write(episode_id, action_episode(4))
    dataset = LazyEpisodeDataset(tmp_path, max_transitions=9)
    assert len(dataset) == 3
    assert dataset.transition_count == 12


def test_lazy_dataset_filters_episodes_without_state_transition(tmp_path):
    writer = EpisodeWriter(tmp_path, DatasetKind.ACTION_ONLY)
    writer.write(0, action_episode(1))
    writer.write(1, action_episode(2))
    dataset = LazyEpisodeDataset(tmp_path, min_sequence_length=2)
    assert len(dataset) == 1
    assert dataset.filtered_episode_count == 1


def test_action_prior_training_preserves_loss_history(tmp_path, monkeypatch):
    data = tmp_path / "action_data"
    writer = EpisodeWriter(data, DatasetKind.ACTION_ONLY)
    for episode_id in range(4):
        episode = action_episode(4)
        episode["actions"] = np.random.default_rng(episode_id).uniform(
            -0.8, 0.8, size=(4, 2)
        ).astype(np.float32)
        writer.write(episode_id, episode)

    output = tmp_path / "action_prior.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_prior",
            "--prior",
            "action",
            "--data",
            str(data),
            "--output",
            str(output),
            "env=windy",
            "env.image_size=[8,8]",
            "device=cpu",
            "epochs=1",
            "batch_size=2",
            "model.action_prior.d_model=16",
            "model.action_prior.n_layers=1",
            "model.action_prior.n_heads=2",
        ],
    )
    train_prior.main()

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert len(checkpoint["history"]["action_prior"]) == 1
    assert np.isfinite(checkpoint["history"]["action_prior"][0])
    assert (tmp_path / "action_prior_diagnostics" / "training_loss.png").exists()


def test_early_stopper_uses_min_delta_warmup_and_patience():
    stopper = _EarlyStopper(patience=2, min_delta=0.1, warmup_epochs=2)
    assert stopper.step(1.0, 1) == (True, False)
    assert stopper.step(0.95, 2) == (False, False)
    assert stopper.step(0.96, 3) == (False, True)
    assert stopper.best_epoch == 1


def test_state_prior_scheduler_early_stop_and_best_restore(tmp_path, monkeypatch):
    data = tmp_path / "state_data"
    writer = EpisodeWriter(data, DatasetKind.STATE_ONLY)
    for episode_id in range(20):
        writer.write(episode_id, state_episode(episode_id))

    token_data = tmp_path / "state_tokens"
    token_writer = EpisodeWriter(token_data, DatasetKind.TOKENS)
    for episode_id in range(20):
        token_writer.write(
            episode_id,
            {
                "video_tokens": np.zeros((4, 1, 1), np.int64),
                "sequence_length": np.asarray(4, np.int64),
            },
        )
    vqvae = tmp_path / "vqvae.pt"
    tokenizer = VQVAE(hidden_dim=16, codebook_size=16, code_dim=8)
    torch.save({"kind": "vqvae", "tokenizer": tokenizer.state_dict()}, vqvae)

    output = tmp_path / "state_prior.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_prior",
            "--prior",
            "state",
            "--data",
            str(data),
            "--output",
            str(output),
            "--tokens",
            str(token_data),
            "--vqvae",
            str(vqvae),
            "env=windy",
            "env.image_size=[8,8]",
            "device=cpu",
            "epochs=10",
            "batch_size=4",
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
            "model.state_prior.training.observation_warmup_epochs=0",
            "model.state_prior.training.video_warmup_epochs=0",
            "model.state_prior.learning_rate=0.001",
            "model.state_prior.scheduler.name=reduce_on_plateau",
            "model.state_prior.scheduler.patience=0",
            "model.state_prior.scheduler.threshold=1000000",
            "model.state_prior.early_stopping.enabled=true",
            "model.state_prior.early_stopping.patience=2",
            "model.state_prior.early_stopping.min_delta=1000000",
            "model.state_prior.early_stopping.warmup_epochs=0",
            "model.state_prior.early_stopping.restore_best=true",
        ],
    )
    train_prior.main()

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    summary = checkpoint["training_summary"]["phases"][-1]
    assert summary["early_stopped"] is True
    assert summary["epochs_completed"] == 3
    assert summary["best_epoch"] == 1
    assert summary["restored_best_weights"] is True
    assert len(checkpoint["history"]["state_prior"]) == 3
    assert summary["final_learning_rate"] < 0.001


class _SuccessfulNonTerminatingEnv:
    action_low = np.full(2, -1, np.float32)
    action_high = np.full(2, 1, np.float32)

    def __init__(self):
        self.step_index = 0

    @staticmethod
    def _observation():
        return ObservationBundle(
            rgb=np.zeros((4, 4, 3), np.uint8),
            proprio=np.zeros(2, np.float32),
            control_state=np.zeros(2, np.float32),
            state_mask=np.ones(2, bool),
            goal=np.zeros(1, np.float32),
        )

    def reset(self, seed=0):
        self.step_index = 0
        return self._observation()

    def step(self, action):
        rewards = (-1.0, 0.0, -1.0, -1.0)
        reward = rewards[self.step_index]
        self.step_index += 1
        return StepResult(self._observation(), reward, False, False, {})


def test_expert_labels_stop_after_first_success():
    episode = generate_episode(
        _SuccessfulNonTerminatingEnv(),
        DatasetKind.PAIRED,
        seed=0,
        horizon=4,
        paired_config={
            "expert_fraction": 1.0,
            "successful_expert_only": True,
            "success_reward_threshold": 0.0,
            "expert_until_first_success": True,
        },
    )
    assert episode["action_is_expert"].tolist() == [True, True, False, False]


def test_paired_schema_accepts_distinct_relabelled_expert_actions():
    length = 3
    episode = {
        "rgb": np.zeros((length, 4, 4, 3), np.uint8),
        "proprio": np.zeros((length, 2), np.float32),
        "control_state": np.zeros((length, 2), np.float32),
        "state_mask": np.ones((length, 2), bool),
        "actions": np.zeros((length, 2), np.float32),
        "expert_actions": np.ones((length, 2), np.float32),
        "next_rgb": np.zeros((length, 4, 4, 3), np.uint8),
        "next_proprio": np.zeros((length, 2), np.float32),
        "next_control_state": np.zeros((length, 2), np.float32),
        "next_state_mask": np.ones((length, 2), bool),
        "rewards": np.zeros(length, np.float32),
        "terminated": np.zeros(length, bool),
        "truncated": np.zeros(length, bool),
        "goals": np.zeros((length, 1), np.float32),
        "sequence_length": np.asarray(length, np.int64),
    }
    validate_episode(DatasetKind.PAIRED, episode)
