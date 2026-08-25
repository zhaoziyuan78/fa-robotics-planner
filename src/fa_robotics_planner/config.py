"""Small YAML composition layer used by every CLI.

Hydra is intentionally not a hard dependency on cluster workers.  The accepted
``group=name`` syntax and dotted overrides mirror the subset used by the
documented commands while keeping every resolved run configuration serializable.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


CONFIG_GROUPS = {"env", "model", "planner", "baseline", "eval", "profile"}


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = cursor.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {path!r}: {part!r} is not a mapping")
        cursor = node
    cursor[parts[-1]] = value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def compose_config(
    overrides: Iterable[str] = (),
    *,
    config_root: str | Path | None = None,
    base: str = "default.yaml",
) -> dict[str, Any]:
    root = Path(config_root or Path(__file__).resolve().parents[2] / "configs")
    config = load_yaml(root / base) if (root / base).exists() else {}
    scalar_overrides: list[tuple[str, Any]] = []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if key in CONFIG_GROUPS and "." not in key:
            group_path = root / key / f"{raw}.yaml"
            if not group_path.exists():
                raise FileNotFoundError(f"Unknown {key} config {raw!r}: {group_path}")
            config = deep_merge(config, load_yaml(group_path))
        else:
            scalar_overrides.append((key, _parse_value(raw)))
    for key, value in scalar_overrides:
        set_dotted(config, key, value)
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)

