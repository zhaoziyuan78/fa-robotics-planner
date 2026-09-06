"""Dataset field contracts that prevent accidental prior information leakage."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

import numpy as np


class DatasetKind(str, Enum):
    STATE_ONLY = "state_prior"
    ACTION_ONLY = "action_prior"
    PAIRED = "paired"
    TOKENS = "tokens"


STATE_ONLY_REQUIRED = {
    "rgb",
    "proprio",
    "control_state",
    "state_mask",
    "sequence_length",
}
ACTION_ONLY_REQUIRED = {"actions", "sequence_length", "action_low", "action_high"}
ACTION_ONLY_ALLOWED = frozenset(ACTION_ONLY_REQUIRED)
PAIRED_REQUIRED = {
    "rgb",
    "proprio",
    "control_state",
    "state_mask",
    "actions",
    "next_rgb",
    "next_proprio",
    "next_control_state",
    "next_state_mask",
    "rewards",
    "terminated",
    "truncated",
    "goals",
    "sequence_length",
}
PAIRED_OPTIONAL_TEMPORAL = {"action_is_expert", "expert_actions"}
TOKENS_REQUIRED = {"video_tokens", "sequence_length"}
TOKENS_OPTIONAL_TEMPORAL = {"next_video_tokens"}


def _length(array: np.ndarray) -> int:
    return int(np.asarray(array).shape[0])


def validate_episode(kind: DatasetKind | str, episode: Mapping[str, np.ndarray]) -> None:
    kind = DatasetKind(kind)
    fields = set(episode)
    required = {
        DatasetKind.STATE_ONLY: STATE_ONLY_REQUIRED,
        DatasetKind.ACTION_ONLY: ACTION_ONLY_REQUIRED,
        DatasetKind.PAIRED: PAIRED_REQUIRED,
        DatasetKind.TOKENS: TOKENS_REQUIRED,
    }[kind]
    missing = required - fields
    if missing:
        raise ValueError(f"{kind.value} episode is missing fields: {sorted(missing)}")
    if kind is DatasetKind.ACTION_ONLY:
        forbidden = fields - ACTION_ONLY_ALLOWED
        if forbidden:
            raise ValueError(
                "Action-only shards may not contain observation/reward/task metadata; "
                f"forbidden fields: {sorted(forbidden)}"
            )
    length = int(np.asarray(episode["sequence_length"]).item())
    temporal = required - {"sequence_length", "action_low", "action_high"}
    if kind is DatasetKind.PAIRED:
        temporal |= fields & PAIRED_OPTIONAL_TEMPORAL
    if kind is DatasetKind.TOKENS:
        temporal |= fields & TOKENS_OPTIONAL_TEMPORAL
    bad = {name: _length(episode[name]) for name in temporal if _length(episode[name]) != length}
    if bad:
        raise ValueError(f"Temporal fields do not match sequence_length={length}: {bad}")
    if kind not in {DatasetKind.ACTION_ONLY, DatasetKind.TOKENS}:
        state = np.asarray(episode["control_state"])
        mask = np.asarray(episode["state_mask"])
        if state.shape != mask.shape:
            raise ValueError("control_state and state_mask shapes differ")
    if kind is DatasetKind.PAIRED:
        if np.asarray(episode["next_control_state"]).shape != np.asarray(episode["next_state_mask"]).shape:
            raise ValueError("next_control_state and next_state_mask shapes differ")
