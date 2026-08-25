from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_planning_fan(
    prior_trajectories: np.ndarray,
    adapted_trajectories: np.ndarray,
    best_trajectory: np.ndarray,
    executed_trajectory: np.ndarray,
    output: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    for trajectory in np.asarray(prior_trajectories):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", alpha=0.08)
    for trajectory in np.asarray(adapted_trajectories):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:orange", alpha=0.08)
    axis.plot(best_trajectory[:, 0], best_trajectory[:, 1], color="red", linewidth=2, label="best predicted")
    axis.plot(executed_trajectory[:, 0], executed_trajectory[:, 1], color="black", linewidth=2, label="executed")
    axis.set_aspect("equal")
    axis.legend()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)

