"""Atomic sharded-NPZ writer with a checksummed manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schemas import DatasetKind, validate_episode


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


class EpisodeWriter:
    def __init__(self, root: str | Path, kind: DatasetKind | str, metadata: Mapping[str, Any] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.kind = DatasetKind(kind)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = {
            "format_version": 1,
            "kind": self.kind.value,
            "metadata": dict(metadata or {}),
            "episodes": [],
        }

    def write(self, episode_id: int, episode: Mapping[str, np.ndarray]) -> Path:
        arrays = {name: np.asarray(value) for name, value in episode.items()}
        validate_episode(self.kind, arrays)
        filename = f"episode_{int(episode_id):06d}.npz"
        target = self.root / filename
        temporary = self.root / f".{filename}.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
        self.manifest["episodes"].append(
            {
                "id": int(episode_id),
                "file": filename,
                "sha256": _digest(target),
                "length": int(arrays["sequence_length"].item()),
                "fields": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in arrays.items()
                },
            }
        )
        self.flush()
        return target

    def flush(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.manifest_path)


def check_dataset(root: str | Path, verify_checksums: bool = True) -> list[str]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return [f"Missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    kind = DatasetKind(manifest["kind"])
    seen: set[int] = set()
    for entry in manifest.get("episodes", []):
        episode_id = int(entry["id"])
        if episode_id in seen:
            errors.append(f"Duplicate episode id: {episode_id}")
        seen.add(episode_id)
        path = root / entry["file"]
        if not path.exists():
            errors.append(f"Missing shard: {path.name}")
            continue
        if verify_checksums and _digest(path) != entry["sha256"]:
            errors.append(f"Checksum mismatch: {path.name}")
        try:
            with np.load(path, allow_pickle=False) as data:
                validate_episode(kind, {name: data[name] for name in data.files})
        except Exception as exc:
            errors.append(f"Invalid shard {path.name}: {exc}")
    return errors

