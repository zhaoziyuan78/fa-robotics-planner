"""Common control/world-model metrics with dependency-free statistics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def bootstrap_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(list(values), np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[indices].mean(axis=1)
    alpha = (1 - float(confidence)) / 2
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1 - alpha)),
        "n": int(values.size),
    }


def summarize_episodes(episodes: Iterable[Mapping[str, Any]], bootstrap_samples: int = 10000) -> dict[str, Any]:
    episodes = list(episodes)
    fields = (
        "return",
        "success",
        "time_to_success",
        "episode_length",
        "control_energy",
        "planning_latency",
        "model_forward_calls",
        "sampled_action_sequences",
        "final_goal_distance",
        "goal_sparse_return",
    )
    result: dict[str, Any] = {"episodes": len(episodes)}
    for field in fields:
        values = [float(episode[field]) for episode in episodes if field in episode and episode[field] is not None]
        if values:
            result[field] = bootstrap_ci(values, samples=bootstrap_samples)
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def candidate_ranking_metrics(predicted: np.ndarray, actual: np.ndarray, top_k: int = 5) -> dict[str, float]:
    predicted, actual = np.asarray(predicted, np.float64), np.asarray(actual, np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 1:
        raise ValueError("predicted and actual returns must be equal-length vectors")
    pr, ar = _rank(predicted), _rank(actual)
    spearman = float(np.corrcoef(pr, ar)[0, 1]) if predicted.size > 1 else 1.0
    concordant = discordant = 0
    for left in range(predicted.size):
        for right in range(left + 1, predicted.size):
            product = (predicted[left] - predicted[right]) * (actual[left] - actual[right])
            concordant += product > 0
            discordant += product < 0
    kendall = (concordant - discordant) / max(1, concordant + discordant)
    predicted_best = int(np.argmax(predicted))
    actual_best = float(actual.max())
    actual_top = set(np.argsort(actual)[-min(top_k, actual.size) :].tolist())
    predicted_top = set(np.argsort(predicted)[-min(top_k, predicted.size) :].tolist())
    return {
        "spearman": spearman,
        "kendall": float(kendall),
        "top1_regret": actual_best - float(actual[predicted_best]),
        "top5_recall": len(actual_top & predicted_top) / max(1, len(actual_top)),
    }
