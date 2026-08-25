from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


FAIRNESS_FIELDS = (
    "environment_steps",
    "paired_steps",
    "episode_horizon",
    "action_repeat",
    "control_frequency",
    "observation_mode",
    "image_size",
    "reward",
    "goal_definition",
    "evaluation_seeds",
    "planning_candidates",
    "planning_horizon",
    "nominal_controller",
    "action_bounds",
)


def fairness_report(methods: Mapping[str, Mapping[str, Any]], output: str | Path) -> list[str]:
    warnings: list[str] = []
    names = list(methods)
    lines = ["# Fairness report", "", "| Field | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for field in FAIRNESS_FIELDS:
        values = [methods[name].get(field, "missing") for name in names]
        lines.append(f"| {field} | " + " | ".join(map(str, values)) + " |")
        normalized = {repr(value) for value in values if value != "missing"}
        if len(normalized) > 1:
            warnings.append(f"Mismatch in {field}: {dict(zip(names, values))}")
    lines.extend(["", "## Additional data disclosure", ""])
    for name, config in methods.items():
        lines.append(
            f"- {name}: state-only frames={config.get('state_only_steps', 0)}, "
            f"action-only steps={config.get('action_only_steps', 0)}, paired/online={config.get('paired_steps', 0)}"
        )
    lines.extend(["", "## Warnings", ""] + ([f"- {warning}" for warning in warnings] or ["None."]))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return warnings

