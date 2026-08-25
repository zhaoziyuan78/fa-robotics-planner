from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fa_robotics_planner.config import compose_config


ROOT = Path(__file__).resolve().parents[1]


def config_from_unknown(unknown: list[str], extra: list[str] = ()) -> dict[str, Any]:
    overrides = [item for item in [*extra, *unknown] if "=" in item]
    return compose_config(overrides, config_root=ROOT / "configs")


def resolve_env_override(name: str | None, unknown: list[str]) -> list[str]:
    if name:
        return [f"env={name}", *unknown]
    return unknown


def data_path(config: dict[str, Any], env_name: str, dataset: str) -> Path:
    """Return the conventional dataset path under the configured data root."""
    return Path(config.get("data_root", "data")).expanduser() / env_name / dataset


def checkpoint_path(config: dict[str, Any], category: str, filename: str) -> Path:
    """Return a model path under the configured checkpoint root."""
    return Path(config.get("checkpoint_root", "checkpoints")).expanduser() / category / filename


def dump(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
