"""Common control/world-model metrics with dependency-free statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def write_episode_metrics(
    path: str | Path, episodes: Iterable[Mapping[str, Any]]
) -> Path:
    """Atomically replace an evaluation JSONL with per-step reward records.

    Baseline workers historically used two filenames and some appended to a
    previous invocation of the same experiment.  Keeping this small writer in
    the dependency-free metrics module gives every evaluator one canonical,
    validated protocol without importing the main experiment runner.
    """

    target = Path(path)
    rows = []
    for index, episode in enumerate(episodes):
        row = dict(episode)
        if "rewards" not in row:
            raise ValueError(
                f"Episode {index} is missing the per-step 'rewards' sequence"
            )
        rewards = [float(reward) for reward in row["rewards"]]
        if not rewards:
            raise ValueError(f"Episode {index} has an empty 'rewards' sequence")
        if not np.isfinite(rewards).all():
            raise ValueError(f"Episode {index} rewards contain NaN or infinity")
        if "episode_length" in row and int(row["episode_length"]) != len(rewards):
            raise ValueError(
                f"Episode {index} length does not match its rewards sequence"
            )
        if "return" in row and not np.isclose(
            float(row["return"]), sum(rewards), rtol=1e-6, atol=1e-6
        ):
            raise ValueError(
                f"Episode {index} return does not equal the sum of rewards"
            )
        row["rewards"] = rewards
        rows.append(row)
    if not rows:
        raise ValueError("Cannot write an empty evaluation metrics file")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


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
