from __future__ import annotations

from typing import Iterable

import numpy as np


def rollout_error(predicted: np.ndarray, target: np.ndarray, horizons: Iterable[int] = (1, 2, 5, 10, 20)) -> dict[int, float]:
    predicted, target = np.asarray(predicted), np.asarray(target)
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("predicted and target rollout arrays must have identical [N,H,...] shape")
    result = {}
    for horizon in horizons:
        if horizon <= predicted.shape[1]:
            difference = predicted[:, horizon - 1] - target[:, horizon - 1]
            result[int(horizon)] = float(np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=-1).mean())
    return result


def intervention_error(
    predicted_action: np.ndarray,
    predicted_passive: np.ndarray,
    actual_action: np.ndarray,
    actual_passive: np.ndarray,
) -> float:
    predicted_delta = np.asarray(predicted_action) - np.asarray(predicted_passive)
    actual_delta = np.asarray(actual_action) - np.asarray(actual_passive)
    difference = predicted_delta - actual_delta
    return float(np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=-1).mean())


def contact_metrics(
    predicted_contact: np.ndarray,
    actual_contact: np.ndarray,
    predicted_velocity: np.ndarray | None = None,
    actual_velocity: np.ndarray | None = None,
) -> dict[str, float]:
    predicted_contact = np.asarray(predicted_contact, bool)
    actual_contact = np.asarray(actual_contact, bool)
    if predicted_contact.shape != actual_contact.shape:
        raise ValueError("Contact arrays must have identical [N,H] shape")
    occurrence = (predicted_contact.any(1) == actual_contact.any(1)).mean()
    sentinel = predicted_contact.shape[1]
    predicted_first = np.where(predicted_contact.any(1), predicted_contact.argmax(1), sentinel)
    actual_first = np.where(actual_contact.any(1), actual_contact.argmax(1), sentinel)
    result = {
        "contact_occurrence_accuracy": float(occurrence),
        "first_contact_time_error": float(np.abs(predicted_first - actual_first).mean()),
    }
    if predicted_velocity is not None and actual_velocity is not None:
        difference = np.asarray(predicted_velocity) - np.asarray(actual_velocity)
        result["post_contact_object_velocity_error"] = float(
            np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=-1).mean()
        )
    return result

