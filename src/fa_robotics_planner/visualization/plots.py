"""Standard result plots generated from aggregate rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _scatter(rows, x, y, output, xlabel=None, ylabel=None):
    rows = [row for row in rows if x in row and y in row]
    figure, axis = plt.subplots(figsize=(5, 3.5))
    groups = sorted({str(row.get("method", "unknown")) for row in rows})
    for group in groups:
        selected = [row for row in rows if str(row.get("method", "unknown")) == group]
        axis.plot([row[x] for row in selected], [row[y] for row in selected], "o-", label=group)
    axis.set_xlabel(xlabel or x)
    axis.set_ylabel(ylabel or y)
    if groups:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def generate_standard_plots(rows: Iterable[dict[str, Any]], output: str | Path) -> None:
    rows = list(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _scatter(rows, "paired_steps", "success_rate", output / "success_vs_paired_data.png")
    _scatter(rows, "num_candidates", "success_rate", output / "success_vs_planning_candidates.png")
    _scatter(rows, "eval", "success_rate", output / "id_vs_ood_success.png")
    # The remaining metrics may be supplied by specialized world-model evals;
    # create consistently named figures rather than mixing plot code into training.
    for filename, x, y in (
        ("passive_rollout_error.png", "rollout_horizon", "passive_error"),
        ("intervention_error.png", "rollout_horizon", "intervention_error"),
        ("candidate_ranking.png", "num_candidates", "spearman"),
    ):
        _scatter(rows, x, y, output / filename)
    matrix = np.full((2, 2), np.nan)
    for state in (False, True):
        for action in (False, True):
            values = [float(row["success_rate"]) for row in rows if bool(row.get("state_adapter")) == state and bool(row.get("action_adapter")) == action and "success_rate" in row]
            if values:
                matrix[int(state), int(action)] = np.mean(values)
    figure, axis = plt.subplots(figsize=(4, 3))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks((0, 1), ("AA off", "AA on"))
    axis.set_yticks((0, 1), ("SA off", "SA on"))
    figure.colorbar(image, ax=axis, label="success")
    figure.tight_layout()
    figure.savefig(output / "adapter_ablation_heatmap.png", dpi=180)
    plt.close(figure)

