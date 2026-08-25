"""Configuration-level OOD validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _outside(value: float, interval: Sequence[float]) -> bool:
    return value < float(interval[0]) or value > float(interval[1])


def validate_ood_config(
    train_ranges: Mapping[str, Sequence[float]],
    conditions: Mapping[str, Mapping[str, Any]],
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
) -> list[str]:
    errors: list[str] = []
    overlap = set(map(int, train_seeds)) & set(map(int, eval_seeds))
    if overlap:
        errors.append(f"Train/eval seeds overlap: {sorted(overlap)}")
    for name, condition in conditions.items():
        changed = False
        relevant = False
        for parameter, value in condition.items():
            if parameter in {"action_space", "observation_schema"}:
                errors.append(f"{name} illegally changes {parameter}")
            if parameter in train_ranges and isinstance(value, (int, float)):
                relevant = True
                changed |= _outside(float(value), train_ranges[parameter])
        if relevant and not changed:
            errors.append(f"{name} has no numeric parameter outside the training range")
    return errors
