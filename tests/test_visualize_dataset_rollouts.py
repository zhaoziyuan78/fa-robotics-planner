import json
from pathlib import Path

import numpy as np

from scripts.visualize_dataset_rollouts import (
    load_episode_frames,
    resolve_dataset,
    select_best_episode,
    uniformly_spaced_indices,
)


def _write_dataset(root: Path, rewards: list[list[float]]) -> Path:
    root.mkdir(parents=True)
    episodes = []
    for episode_id, episode_rewards in enumerate(rewards):
        length = len(episode_rewards)
        name = f"episode_{episode_id:06d}.npz"
        rgb = np.full((length, 4, 5, 3), episode_id, dtype=np.uint8)
        next_rgb = np.full((length, 4, 5, 3), episode_id + 10, dtype=np.uint8)
        np.savez_compressed(
            root / name,
            rewards=np.asarray(episode_rewards, dtype=np.float32),
            rgb=rgb,
            next_rgb=next_rgb,
        )
        episodes.append({"id": episode_id, "file": name, "length": length})
    (root / "manifest.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )
    return root


def test_fetch_selection_prefers_sustained_final_success(tmp_path):
    dataset = _write_dataset(
        tmp_path / "paired",
        [[-1.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
    )

    selection = select_best_episode(dataset, "fetch_push")

    assert selection.episode_id == 1
    assert selection.success_steps == 2
    assert selection.final_success is True
    frames = load_episode_frames(selection)
    assert frames.shape == (4, 4, 5, 3)
    assert np.all(frames[-1] == 11)


def test_humanoid_selection_rejects_one_step_reward_outlier(tmp_path):
    dataset = _write_dataset(
        tmp_path / "paired",
        [[1000.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
    )

    selection = select_best_episode(dataset, "humanoid_push")

    assert selection.episode_id == 2
    assert selection.minimum_valid_length == 2


def test_dataset_resolution_prefers_newest_available_paper_variant(tmp_path):
    paper_v2 = tmp_path / "paper_v2" / "humanoid_stand" / "paired"
    paper_v4 = tmp_path / "paper_v4" / "humanoid_stand" / "paired"
    _write_dataset(paper_v2, [[1.0, 1.0]])
    assert resolve_dataset(tmp_path, "humanoid_stand") == paper_v2
    _write_dataset(paper_v4, [[1.0, 1.0]])
    assert resolve_dataset(tmp_path, "humanoid_stand") == paper_v4


def test_uniform_sampling_covers_the_entire_rollout():
    indices = uniformly_spaced_indices(501, 160)

    assert len(indices) == 160
    assert indices[0] == 0
    assert indices[-1] == 500
    assert np.all(np.diff(indices) > 0)
