"""Compact training diagnostics shared by priors and adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def save_training_history(
    history: Mapping[str, Sequence[float]], output_prefix: str | Path
) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(name): [float(value) for value in values]
        for name, values in history.items()
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez(prefix.with_suffix(".npz"), **{k: np.asarray(v) for k, v in serializable.items()})
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in serializable.items():
        if values and name not in {"learning_rate", "kl_weight"}:
            axis.plot(np.arange(1, len(values) + 1), values, label=name)
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.2)
    if axis.lines:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(figure)


def save_action_comparison(
    expert: np.ndarray,
    prior: np.ndarray,
    adapted: np.ndarray,
    output: str | Path,
) -> None:
    expert = np.asarray(expert)
    prior = np.asarray(prior)
    adapted = np.asarray(adapted)
    if expert.shape != prior.shape or expert.shape != adapted.shape or expert.ndim != 2:
        raise ValueError("Action comparison arrays must share shape (time, action_dim)")
    dimensions = expert.shape[1]
    shown = min(dimensions, 12)
    figure, axes = plt.subplots(shown, 1, figsize=(8, max(3, 1.35 * shown)), sharex=True)
    axes = np.atleast_1d(axes)
    time = np.arange(expert.shape[0])
    for dimension, axis in enumerate(axes):
        axis.plot(time, expert[:, dimension], color="black", linewidth=1.3, label="expert")
        axis.plot(time, prior[:, dimension], color="tab:blue", alpha=0.85, label="prior mean")
        axis.plot(time, adapted[:, dimension], color="tab:orange", alpha=0.9, label="adapted mean")
        axis.set_ylabel(f"a{dimension}")
        axis.grid(alpha=0.15)
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("transition")
    if dimensions > shown:
        figure.suptitle(f"First {shown}/{dimensions} action dimensions")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)


def save_state_prediction_comparison(
    actual: np.ndarray,
    passive: np.ndarray,
    adapted: np.ndarray,
    mask: np.ndarray,
    output: str | Path,
    fps: int = 8,
    max_dimensions: int = 16,
) -> None:
    actual = np.asarray(actual, np.float32)
    passive = np.asarray(passive, np.float32)
    adapted = np.asarray(adapted, np.float32)
    mask = np.asarray(mask, bool)
    if not (actual.shape == passive.shape == adapted.shape == mask.shape):
        raise ValueError("State comparison arrays and mask must have identical shapes")
    valid_dimensions = np.flatnonzero(mask.any(axis=0))
    if not valid_dimensions.size:
        raise ValueError("State comparison has no valid dimensions")
    error = np.full(actual.shape[1], -np.inf, np.float32)
    for dimension in valid_dimensions:
        selected_time = mask[:, dimension]
        error[dimension] = np.mean(
            np.abs(passive[selected_time, dimension] - actual[selected_time, dimension])
        )
    order = valid_dimensions[np.argsort(np.nan_to_num(error[valid_dimensions]))[::-1]]
    selected = np.sort(order[: int(max_dimensions)])
    frames: list[np.ndarray] = []
    for timestep in range(actual.shape[0]):
        figure, axis = plt.subplots(figsize=(7, 3.6))
        x = np.arange(selected.size)
        axis.plot(x, actual[timestep, selected], "o-", color="black", label="simulator")
        axis.plot(x, passive[timestep, selected], "o-", color="tab:blue", label="State Prior")
        axis.plot(x, adapted[timestep, selected], "o-", color="tab:orange", label="+ State Adapter")
        current_mask = mask[timestep, selected]
        passive_rmse = np.sqrt(np.mean((passive[timestep, selected][current_mask] - actual[timestep, selected][current_mask]) ** 2))
        adapted_rmse = np.sqrt(np.mean((adapted[timestep, selected][current_mask] - actual[timestep, selected][current_mask]) ** 2))
        axis.set_title(
            f"one-step state prediction t={timestep} | prior RMSE={passive_rmse:.4f} | adapted={adapted_rmse:.4f}"
        )
        axis.set_xticks(x, [str(index) for index in selected], rotation=45)
        axis.set_xlabel("state dimension")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(figure)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(target, np.stack(frames), fps=int(fps))
