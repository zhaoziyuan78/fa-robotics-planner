"""Shared, read-only paired-data protocol for every offline baseline.

The baseline trainers deliberately consume the same public transition fields.
No environment object is constructed by this module, which makes it possible
to assert that training used zero environment interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class OfflineEpisode:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    goals: np.ndarray
    rgb: np.ndarray | None = None
    next_rgb: np.ndarray | None = None
    action_is_expert: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class OfflinePairedData:
    root: Path
    episodes: tuple[OfflineEpisode, ...]
    requested_transitions: int
    manifest_transitions: int

    @property
    def transition_count(self) -> int:
        return sum(episode.length for episode in self.episodes)

    @property
    def state_dim(self) -> int:
        return int(self.episodes[0].states.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0].actions.shape[-1])

    @property
    def goal_dim(self) -> int:
        return int(self.episodes[0].goals.shape[-1])

    def transitions(self) -> Iterator[tuple[np.ndarray, ...]]:
        for episode in self.episodes:
            for index in range(episode.length):
                yield (
                    episode.states[index],
                    episode.actions[index],
                    episode.next_states[index],
                    episode.rewards[index],
                    episode.dones[index],
                    episode.goals[index],
                )


def resolve_paired_data_path(
    data_root: str | Path, environment: str, explicit: str | Path | None = None
) -> Path:
    """Resolve one paired dataset without silently changing data domains."""

    path = Path(explicit).expanduser() if explicit else Path(data_root).expanduser() / environment / "paired"
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Offline paired dataset not found: {manifest}. Generate it first or "
            "set baseline.data=/absolute/path/to/paired."
        )
    return path.resolve()


def load_paired_data(
    root: str | Path,
    max_transitions: int,
    *,
    include_rgb: bool = False,
) -> OfflinePairedData:
    """Load exactly ``max_transitions`` (or fail), preserving episode order."""

    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "paired":
        raise ValueError(f"Baseline data must be paired, got {manifest.get('kind')!r}")
    requested = int(max_transitions)
    if requested <= 0:
        raise ValueError("offline_transitions must be positive")
    entries = sorted(manifest.get("episodes", []), key=lambda item: int(item["id"]))
    available = sum(int(entry["length"]) for entry in entries)
    if available < requested:
        raise ValueError(
            f"Requested {requested} offline transitions but {root} contains only {available}"
        )

    episodes: list[OfflineEpisode] = []
    remaining = requested
    expected_shapes: tuple[int, int, int] | None = None
    for entry in entries:
        if remaining <= 0:
            break
        with np.load(root / entry["file"], allow_pickle=False) as shard:
            count = min(remaining, int(np.asarray(shard["sequence_length"]).item()))
            states = np.asarray(shard["control_state"][:count], np.float32).copy()
            actions = np.asarray(shard["actions"][:count], np.float32).copy()
            next_states = np.asarray(shard["next_control_state"][:count], np.float32).copy()
            rewards = np.asarray(shard["rewards"][:count], np.float32).reshape(-1).copy()
            terminated = np.asarray(shard["terminated"][:count], bool).reshape(-1)
            truncated = np.asarray(shard["truncated"][:count], bool).reshape(-1)
            goals = np.asarray(shard["goals"][:count], np.float32).copy()
            masks = np.asarray(shard["state_mask"][:count], bool)
            next_masks = np.asarray(shard["next_state_mask"][:count], bool)
            # Padded coordinates are public schema capacity, not observations.
            states[~masks] = 0.0
            next_states[~next_masks] = 0.0
            rgb = np.asarray(shard["rgb"][:count], np.uint8).copy() if include_rgb else None
            next_rgb = (
                np.asarray(shard["next_rgb"][:count], np.uint8).copy()
                if include_rgb
                else None
            )
            expert = (
                np.asarray(shard["action_is_expert"][:count], bool).reshape(-1).copy()
                if "action_is_expert" in shard.files
                else None
            )
        shapes = (states.shape[-1], actions.shape[-1], goals.shape[-1])
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            raise ValueError(
                f"Inconsistent state/action/goal dimensions: {shapes} != {expected_shapes}"
            )
        arrays = (states, actions, next_states, rewards, goals)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"NaN or infinity in paired shard {entry['file']}")
        episodes.append(
            OfflineEpisode(
                states=states,
                actions=actions,
                next_states=next_states,
                rewards=rewards,
                dones=np.logical_or(terminated, truncated).astype(np.float32),
                goals=goals,
                rgb=rgb,
                next_rgb=next_rgb,
                action_is_expert=expert,
            )
        )
        remaining -= count

    result = OfflinePairedData(root, tuple(episodes), requested, available)
    if result.transition_count != requested:
        raise RuntimeError(
            f"Offline loader returned {result.transition_count}, expected {requested}"
        )
    return result


def stack_transitions(data: OfflinePairedData) -> dict[str, np.ndarray]:
    """Stack vector fields only; image users should concatenate explicitly."""

    names = ("states", "actions", "next_states", "rewards", "dones", "goals")
    return {
        name: np.concatenate([getattr(episode, name) for episode in data.episodes], axis=0)
        for name in names
    }


def achieved_goal_slice(config: dict, goal_dim: int) -> slice | None:
    """Return the public achieved-goal coordinates used for offline relabeling."""

    if goal_dim == 0:
        return None
    environment = config.get("env", config)
    bounds = environment.get(
        "action_adapter_achieved_goal_slice", environment.get("achieved_goal_slice")
    )
    if bounds is None:
        raise ValueError("A goal-conditioned offline baseline needs achieved_goal_slice")
    start, stop = map(int, bounds)
    if stop - start < goal_dim:
        raise ValueError("achieved_goal_slice is smaller than goal_size")
    return slice(start, start + goal_dim)
