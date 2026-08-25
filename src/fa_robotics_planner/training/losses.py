from __future__ import annotations

from collections.abc import Callable

import torch

from fa_robotics_planner.models.distributions import ActionDistribution
from fa_robotics_planner.models.distributions import TanhNormal


def action_prior_loss(distribution: ActionDistribution, actions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    log_prob = distribution.log_prob(actions)
    weights = valid.to(log_prob.dtype)
    return -(log_prob * weights).sum() / weights.sum().clamp_min(1)


def tanh_normal_kl(adapted: TanhNormal, prior: TanhNormal) -> torch.Tensor:
    """KL(adapted || prior), invariant to their shared tanh/affine transform."""

    adapted_var = (2.0 * adapted.log_scale).exp()
    prior_var = (2.0 * prior.log_scale).exp()
    per_dimension = (
        prior.log_scale
        - adapted.log_scale
        + (adapted_var + (adapted.loc - prior.loc).square()) / (2.0 * prior_var)
        - 0.5
    )
    return per_dimension.sum(dim=-1)


def soft_action_adapter_loss(
    adapted: TanhNormal,
    prior: TanhNormal,
    targets: torch.Tensor,
    *,
    target_sigma: float = 0.1,
    target_samples: int = 4,
    label_smoothing: float = 0.02,
    likelihood_clip: float = 1e-3,
    kl_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuous counterpart of the legacy Gaussian soft-token objective.

    Targets are convolved with a small Gaussian in normalized action units.
    Tiny uniform label smoothing prevents the adapter variance from collapsing,
    while conservative KL keeps the learned policy near the frozen prior.
    """

    if targets.shape != adapted.loc.shape:
        raise ValueError("Action targets must match the adapter batch shape")
    if target_sigma < 0 or target_samples <= 0:
        raise ValueError("target_sigma must be non-negative and target_samples positive")
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must lie in [0, 1)")
    if likelihood_clip < 0 or likelihood_clip >= 0.5:
        raise ValueError("likelihood_clip must lie in [0, 0.5)")
    action_range = adapted.high - adapted.low
    lower = adapted.low + float(likelihood_clip) * action_range
    upper = adapted.high - float(likelihood_clip) * action_range
    clipped = torch.maximum(torch.minimum(targets, upper), lower)
    if target_sigma > 0:
        noise = torch.randn(
            (int(target_samples),) + tuple(targets.shape),
            dtype=targets.dtype,
            device=targets.device,
        )
        noisy_targets = clipped.unsqueeze(0) + float(target_sigma) * noise
        noisy_targets = torch.maximum(
            torch.minimum(noisy_targets, upper), lower
        )
    else:
        noisy_targets = clipped.unsqueeze(0).expand(
            int(target_samples), *clipped.shape
        )
    soft_nll = -adapted.log_prob(noisy_targets).mean()
    if label_smoothing:
        uniform_targets = lower + torch.rand_like(noisy_targets) * (upper - lower)
        uniform_nll = -adapted.log_prob(uniform_targets).mean()
        soft_nll = (
            (1.0 - float(label_smoothing)) * soft_nll
            + float(label_smoothing) * uniform_nll
        )
    conservative_kl = tanh_normal_kl(adapted, prior).mean()
    total = soft_nll + float(kl_weight) * conservative_kl
    return total, soft_nll, conservative_kl


def masked_state_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dimension_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("predicted, target, and mask must have identical shapes")
    selected = mask.bool()
    if not selected.any():
        # Preserve a differentiable zero without reading potentially non-finite
        # values at invalid/padded positions.
        return predicted.reshape(-1)[:0].sum()
    squared_error = (predicted - target).square()
    if dimension_weights is None:
        return squared_error[selected].mean()
    weights = torch.as_tensor(
        dimension_weights, dtype=predicted.dtype, device=predicted.device
    )
    if weights.ndim != 1 or weights.numel() != predicted.shape[-1]:
        raise ValueError("dimension_weights must have one value per state dimension")
    effective = selected.to(predicted.dtype) * weights
    return (squared_error * effective).sum() / effective.sum().clamp_min(1)


def adapter_rollout_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    lambda_multi: float = 1.0,
    discount: float = 0.99,
) -> torch.Tensor:
    if not predictions or not (len(predictions) == len(targets) == len(masks)):
        raise ValueError("Predictions, targets, and masks must have the same non-zero length")
    loss = masked_state_loss(predictions[0], targets[0], masks[0])
    for horizon in range(1, len(predictions)):
        loss = loss + float(lambda_multi) * float(discount) ** horizon * masked_state_loss(
            predictions[horizon], targets[horizon], masks[horizon]
        )
    return loss
