"""Lazy per-shard dataset and deterministic episode-level splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np


def deterministic_split(
    episode_ids: Iterable[int],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[int]]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to one")
    result = {"train": [], "val": [], "test": []}
    first, second = fractions[0], fractions[0] + fractions[1]
    for episode_id in sorted(map(int, episode_ids)):
        digest = hashlib.sha256(f"{int(seed)}:{episode_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        split = "train" if value < first else "val" if value < second else "test"
        result[split].append(episode_id)
    return result


class LazyEpisodeDataset:
    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        seed: int = 0,
        max_transitions: int | None = None,
        min_sequence_length: int | None = None,
    ):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        entries = sorted(manifest.get("episodes", []), key=lambda entry: int(entry["id"]))
        original_episode_count = len(entries)
        if min_sequence_length is not None:
            minimum = int(min_sequence_length)
            if minimum <= 0:
                raise ValueError("min_sequence_length must be positive")
            entries = [entry for entry in entries if int(entry["length"]) >= minimum]
        self.filtered_episode_count = original_episode_count - len(entries)
        if max_transitions is not None:
            limit = int(max_transitions)
            if limit <= 0:
                raise ValueError("max_transitions must be positive")
            selected = []
            transitions = 0
            for entry in entries:
                if transitions >= limit:
                    break
                selected.append(entry)
                transitions += int(entry["length"])
            entries = selected
        if split is not None:
            partitions = deterministic_split((entry["id"] for entry in entries), seed)
            selected = set(partitions[split])
            entries = [entry for entry in entries if int(entry["id"]) in selected]
        self.entries = entries
        self.kind = manifest["kind"]
        self.transition_count = sum(int(entry["length"]) for entry in entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        path = self.root / self.entries[index]["file"]
        with np.load(path, allow_pickle=False, mmap_mode="r") as data:
            return {name: np.asarray(data[name]) for name in data.files}

    def __iter__(self) -> Iterator[Mapping[str, np.ndarray]]:
        for index in range(len(self)):
            yield self[index]
