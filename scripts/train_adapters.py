from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.models.distributions import TanhNormal
from fa_robotics_planner.training.losses import (
    masked_state_loss,
    soft_action_adapter_loss,
)
from fa_robotics_planner.training.parameters import make_adapter_optimizer, parameter_report
from fa_robotics_planner.utils import seed_everything
from fa_robotics_planner.visualization import (
    save_action_comparison,
    save_state_prediction_comparison,
    save_training_history,
)

from ._common import checkpoint_path, config_from_unknown, data_path
from .train_prior import _StateTokenDataset


def _bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}


def _load_priors(method, state_path: Path, action_path: Path) -> None:
    state_checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state_checkpoint.get("architecture_version", 0)) != int(
        method.state_prior.architecture_version
    ):
        raise ValueError(
            f"State Prior checkpoint {state_path} predates the discrete dual-prior "
            "architecture. Retrain VQ-VAE, tokenize frames, and retrain State Prior."
        )
    state = state_checkpoint["state"]
    method.state_prior.load_state_dict(state["state_prior"])
    if method.tokenizer is None or state.get("tokenizer") is None:
        raise ValueError(f"State Prior checkpoint {state_path} has no bundled VQ-VAE")
    method.tokenizer.load_state_dict(state["tokenizer"])
    action_checkpoint = torch.load(action_path, map_location="cpu", weights_only=False)
    method.action_prior.load_state_dict(action_checkpoint["state"])
    method.freeze_priors()


def _multistep_state_loss(
    method,
    states,
    masks,
    next_states,
    next_masks,
    actions,
    video_tokens,
    next_video_tokens,
    horizon,
    discount,
    start=0,
    context_length=8,
    dimension_weights=None,
):
    """One sampled autoregressive rollout; no future true state enters context."""
    start = int(start)
    horizon = min(int(horizon), states.size(0) - start)
    if horizon < 2:
        return states.new_zeros(())
    context_start = max(0, start + 1 - max(1, int(context_length)))
    # Keep only observations available at the rollout origin.  Starting from
    # a single isolated frame makes latent disturbances (for example Windy
    # region wind) unidentifiable and does not match evaluation, which keeps a
    # short true-observation prefix before every imagined rollout.
    state_context = states[context_start : start + 1].unsqueeze(0)
    mask_context = masks[context_start : start + 1].unsqueeze(0)
    video_context = video_tokens[context_start : start + 1].unsqueeze(0)
    total = states.new_zeros(())
    for offset in range(horizon):
        valid = torch.ones(1, state_context.size(1), dtype=torch.bool, device=states.device)
        prior = method.state_prior.encode_context(
            state_context,
            mask_context,
            video_context,
            valid,
        )
        predicted, _, condition = method.state_adapter(
            state_context[:, -1],
            prior.passive_next,
            actions[start + offset : start + offset + 1],
            prior.observation_hidden,
            prior.video_summary,
        )
        if offset >= 1:
            total = total + float(discount) ** offset * masked_state_loss(
                predicted,
                next_states[start + offset : start + offset + 1],
                next_masks[start + offset : start + offset + 1],
                dimension_weights,
            )
        state_context = torch.cat((state_context, predicted[:, None]), 1)
        mask_context = torch.cat(
            (
                mask_context,
                next_masks[start + offset : start + offset + 1].unsqueeze(0),
            ),
            1,
        )
        def adapt(logits, token_hidden, spatial_index):
            return method.state_adapter.adapt_video_logits(
                logits, token_hidden, spatial_index, condition
            )[0]

        predicted_tokens, _, _ = method.state_prior.generate_next_video(
            prior.video_cache,
            prior.observation_hidden,
            logit_adapter=adapt,
        )
        video_context = torch.cat((video_context, predicted_tokens[:, None]), 1)
    return total


def _state_dimension_weights(config, device: torch.device) -> torch.Tensor:
    """Upweight action-sensitive fields without changing the loss target."""

    state_dim = int(config["env"]["state_size"])
    weights = torch.ones(state_dim, device=device)
    for item in config["env"].get("state_adapter_loss_weights", []):
        bounds = item["slice"]
        start, stop = int(bounds[0]), int(bounds[1])
        if not 0 <= start < stop <= state_dim:
            raise ValueError(f"Invalid state_adapter_loss_weights slice: {bounds}")
        weight = float(item["weight"])
        if weight <= 0:
            raise ValueError("State Adapter dimension weights must be positive")
        weights[start:stop] = weight
    return weights


@torch.inference_mode()
def _save_adapter_diagnostics(
    method,
    episode: dict[str, np.ndarray],
    token_episode: dict[str, np.ndarray] | None,
    device: torch.device,
    output_directory: Path,
    state_enabled: bool,
    action_enabled: bool,
    paired_config: dict[str, object] | None = None,
) -> dict[str, float]:
    """Compare frozen-prior and adapted predictions on one untouched sequence."""

    method.eval()
    maximum = min(
        len(episode["actions"]),
        int(method.action_prior.max_length),
        int(method.state_prior.context_frames),
    )
    if maximum <= 0:
        return {}
    states = torch.as_tensor(episode["control_state"][:maximum], device=device)
    masks = torch.as_tensor(episode["state_mask"][:maximum], device=device)
    actions = torch.as_tensor(episode["actions"][:maximum], device=device)
    goals = torch.as_tensor(episode["goals"][:maximum], device=device)
    metrics: dict[str, float] = {}
    output_directory.mkdir(parents=True, exist_ok=True)
    if state_enabled:
        if token_episode is None:
            current_tokens = method.tokenizer.encode(
                torch.as_tensor(episode["rgb"][:maximum], device=device)
            )
            next_tokens = method.tokenizer.encode(
                torch.as_tensor(episode["next_rgb"][:maximum], device=device)
            )
        else:
            current_tokens = torch.as_tensor(
                token_episode["video_tokens"][:maximum], device=device
            )
            next_tokens = torch.as_tensor(
                token_episode["next_video_tokens"][:maximum], device=device
            )
        actual = torch.as_tensor(
            episode["next_control_state"][:maximum], device=device
        )
        state_sequence = torch.cat((states, actual[-1:]), 0)
        mask_sequence = torch.cat(
            (
                masks,
                torch.as_tensor(
                    episode["next_state_mask"][maximum - 1 : maximum],
                    device=device,
                ),
            ),
            0,
        )
        video_sequence = torch.cat((current_tokens, next_tokens[-1:]), 0)
        valid = torch.ones(1, maximum + 1, dtype=torch.bool, device=device)
        prior = method.state_prior(
            state_sequence.unsqueeze(0),
            mask_sequence.unsqueeze(0),
            video_sequence.unsqueeze(0),
            valid,
        )
        passive = prior.passive_next.squeeze(0)
        adapted_state, _, condition = method.state_adapter(
            states,
            passive,
            actions,
            prior.hidden.squeeze(0),
            prior.video_summary[:, :-1].squeeze(0),
        )
        adapted_logits, _ = method.state_adapter.adapt_video_sequence(
            prior.video_logits.squeeze(0),
            prior.video_predictor_hidden.squeeze(0),
            condition,
        )
        next_mask = torch.as_tensor(
            episode["next_state_mask"][:maximum], device=device
        ).bool()
        passive_error = (passive - actual)[next_mask]
        adapted_error = (adapted_state - actual)[next_mask]
        metrics["state_prior_rmse"] = float(passive_error.square().mean().sqrt())
        metrics["state_adapter_rmse"] = float(adapted_error.square().mean().sqrt())
        metrics["video_prior_token_accuracy"] = float(
            (prior.video_logits.squeeze(0).argmax(-1) == next_tokens.flatten(-2)).float().mean()
        )
        metrics["video_adapter_token_accuracy"] = float(
            (adapted_logits.argmax(-1) == next_tokens.flatten(-2)).float().mean()
        )
        save_state_prediction_comparison(
            actual.float().cpu().numpy(),
            passive.float().cpu().numpy(),
            adapted_state.float().cpu().numpy(),
            next_mask.cpu().numpy(),
            output_directory / "state_adapter_comparison.gif",
        )
    if action_enabled:
        prior = method.action_prior(actions[:-1].unsqueeze(0))
        base = TanhNormal(
            prior.distribution.loc.squeeze(0),
            prior.distribution.log_scale.squeeze(0),
            prior.distribution.low,
            prior.distribution.high,
        )
        adapted_action, _, _ = method.action_adapter(
            base,
            prior.hidden.squeeze(0),
            states,
            goals,
        )
        expert = torch.as_tensor(
            episode.get("expert_actions", episode["actions"])[:maximum],
            device=device,
        )
        expert_mask = torch.as_tensor(
            episode.get(
                "action_is_expert", np.ones(len(episode["actions"]), bool)
            )[:maximum],
            dtype=torch.bool,
            device=device,
        )
        paired_config = dict(paired_config or {})
        if bool(paired_config.get("successful_expert_only", False)):
            successful_steps = np.flatnonzero(
                np.asarray(episode["rewards"][:maximum])
                >= float(paired_config.get("success_reward_threshold", 0.0))
            )
            if not successful_steps.size:
                expert_mask.zero_()
            elif bool(paired_config.get("expert_until_first_success", False)):
                expert_mask[int(successful_steps[0]) + 1 :] = False
        if not expert_mask.any():
            expert_mask.fill_(True)
        expert = expert[expert_mask]
        prior_mean = base.mean[expert_mask]
        adapted_mean = adapted_action.mean[expert_mask]
        metrics["action_prior_mean_rmse"] = float(
            (prior_mean - expert).square().mean().sqrt()
        )
        metrics["action_adapter_mean_rmse"] = float(
            (adapted_mean - expert).square().mean().sqrt()
        )
        save_action_comparison(
            expert.float().cpu().numpy(),
            prior_mean.float().cpu().numpy(),
            adapted_mean.float().cpu().numpy(),
            output_directory / "action_adapter_comparison.png",
        )
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--tokens")
    parser.add_argument("--state-prior")
    parser.add_argument("--action-prior")
    parser.add_argument(
        "--init-adapters",
        help="Optional adapter checkpoint to fine-tune (for compact relabelled data)",
    )
    parser.add_argument("--output")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", *unknown])
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    env_name = config["env"]["name"]
    data = Path(args.data or config.get("data", data_path(config, env_name, "paired")))
    raw_dataset = LazyEpisodeDataset(
        data,
        split="train",
        seed=int(config.get("seed", 0)),
        max_transitions=config.get("max_transitions"),
    )
    print(f"Using {len(raw_dataset)} train episodes ({raw_dataset.transition_count} transitions)")
    sample = raw_dataset[0]
    config["env"]["goal_size"] = int(sample["goals"].shape[-1])
    state_enabled = _bool(config.get("state_adapter", config["model"]["state_adapter"].get("enabled", True)))
    action_enabled = _bool(config.get("action_adapter", config["model"]["action_adapter"].get("enabled", True)))
    config["model"]["state_adapter"]["enabled"] = state_enabled
    config["model"]["action_adapter"]["enabled"] = action_enabled
    token_root = Path(
        args.tokens or data_path(config, env_name, "tokens/paired")
    )
    token_dataset = (
        LazyEpisodeDataset(
            token_root,
            split="train",
            seed=int(config.get("seed", 0)),
            max_transitions=config.get("max_transitions"),
        )
        if state_enabled
        else None
    )
    dataset = (
        _StateTokenDataset(raw_dataset, token_dataset)
        if token_dataset is not None
        else raw_dataset
    )
    method = build_method(config)
    state_path = Path(
        args.state_prior
        or config.get(
            "state_prior_checkpoint",
            checkpoint_path(config, "priors", f"{env_name}_state_prior.pt"),
        )
    )
    action_path = Path(
        args.action_prior
        or config.get(
            "action_prior_checkpoint",
            checkpoint_path(config, "priors", f"{env_name}_action_prior.pt"),
        )
    )
    _load_priors(method, state_path, action_path)
    if args.init_adapters:
        initial = torch.load(
            args.init_adapters, map_location="cpu", weights_only=False
        )
        if state_enabled and initial.get("state_adapter") is not None:
            method.state_adapter.load_state_dict(initial["state_adapter"])
        if action_enabled and initial.get("action_adapter") is not None:
            method.action_adapter.load_state_dict(initial["action_adapter"])
    method.apply_ablation()
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    method.to(device)
    started = time.perf_counter()
    report = parameter_report(method)
    print(json.dumps(report, indent=2))
    optimizer = (
        make_adapter_optimizer(
            method,
            float(config.get("learning_rate", 1e-3)),
            state_learning_rate=float(
                config["model"]["state_adapter"].get(
                    "learning_rate", config.get("learning_rate", 1e-3)
                )
            ),
            action_learning_rate=float(
                config["model"]["action_adapter"].get(
                    "learning_rate", config.get("learning_rate", 1e-3)
                )
            ),
        )
        if report["trainable_parameters"]
        else None
    )
    epochs = int(config.get("epochs", 1))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if optimizer is not None
        else None
    )
    state_gradient_clip_norm = float(
        config["model"]["state_adapter"].get("gradient_clip_norm", 1.0)
    )
    action_gradient_clip_norm = float(
        config["model"]["action_adapter"].get("gradient_clip_norm", 1.0)
    )
    state_parameters = [
        parameter
        for parameter in method.state_adapter.parameters()
        if parameter.requires_grad
    ]
    action_parameters = [
        parameter
        for parameter in method.action_adapter.parameters()
        if parameter.requires_grad
    ]
    state_dimension_weights = _state_dimension_weights(config, device)
    history: dict[str, list[float]] = {
        "state_adapter": [],
        "video_adapter": [],
        "action_adapter": [],
        "action_soft_nll": [],
        "action_conservative_kl": [],
        "kl_weight": [],
        "learning_rate": [],
    }
    action_cfg = config["model"]["action_adapter"]
    kl_start = float(action_cfg.get("kl_lambda", 0.1))
    kl_final = float(action_cfg.get("kl_lambda_final", kl_start))
    kl_anneal_epochs = max(1, int(action_cfg.get("kl_anneal_epochs", epochs)))
    for epoch in range(epochs):
        total_state = total_video = total_action = total_soft_nll = total_action_kl = 0.0
        steps = state_updates = action_updates = 0
        kl_progress = (
            1.0
            if kl_anneal_epochs <= 1
            else min(float(epoch) / float(kl_anneal_epochs - 1), 1.0)
        )
        kl_weight = kl_start + kl_progress * (kl_final - kl_start)
        epoch_rng = np.random.default_rng(seed + epoch)
        episode_indices = (
            epoch_rng.permutation(len(dataset)) if optimizer is not None else ()
        )
        for episode_index in episode_indices:
            loaded_episode = dataset[int(episode_index)]
            if state_enabled:
                episode, token_episode = loaded_episode
            else:
                episode, token_episode = loaded_episode, None
            states = torch.as_tensor(episode["control_state"], device=device)
            masks = torch.as_tensor(episode["state_mask"], device=device)
            next_states = torch.as_tensor(episode["next_control_state"], device=device)
            next_masks = torch.as_tensor(episode["next_state_mask"], device=device)
            actions = torch.as_tensor(episode["actions"], device=device)
            goals = torch.as_tensor(episode["goals"], device=device)
            paired_config = config["env"].get("paired_data", {})
            if "action_is_expert" in episode:
                expert_mask = episode["action_is_expert"]
            elif "legacy_expert_action_rms_threshold" in paired_config:
                # Early Humanoid manifests predate explicit expert labels and
                # in practice contain only OU exploration.  Treating every
                # such action as expert silently teaches the Action Adapter a
                # random policy.  The optional threshold gives legacy datasets
                # a conservative migration path; if nothing qualifies there
                # are intentionally no action-adapter updates.
                action_rms = float(np.sqrt(np.mean(np.square(episode["actions"]))))
                is_expert = action_rms <= float(
                    paired_config["legacy_expert_action_rms_threshold"]
                )
                expert_mask = np.full(len(episode["actions"]), is_expert, bool)
            else:
                expert_mask = np.ones(len(episode["actions"]), bool)
            action_is_expert = torch.as_tensor(
                expert_mask,
                dtype=torch.bool,
                device=device,
            )
            if bool(paired_config.get("successful_expert_only", False)):
                successful_steps = np.flatnonzero(
                    np.asarray(episode["rewards"])
                    >= float(paired_config.get("success_reward_threshold", 0.0))
                )
                if not successful_steps.size:
                    action_is_expert.zero_()
                elif bool(paired_config.get("expert_until_first_success", False)):
                    action_is_expert[int(successful_steps[0]) + 1 :] = False
            if not state_enabled and not action_is_expert.any():
                continue
            video_tokens = (
                torch.as_tensor(token_episode["video_tokens"], device=device)
                if token_episode is not None
                else None
            )
            next_video_tokens = (
                torch.as_tensor(token_episode["next_video_tokens"], device=device)
                if token_episode is not None
                else None
            )
            length = states.size(0)
            valid = torch.ones(1, length, dtype=torch.bool, device=device)
            state_start, state_stop = 0, length
            if state_enabled:
                state_window = min(
                    length,
                    int(config["model"]["state_adapter"].get("train_context_length", 8)),
                )
                if length > state_window:
                    state_start = int(epoch_rng.integers(length - state_window + 1))
                state_stop = state_start + state_window
            action_start, action_stop = 0, length
            action_history = actions[:-1]
            action_prefix = 0
            if action_enabled:
                action_cfg = config["model"]["action_adapter"]
                if length > method.action_prior.max_length:
                    action_stop = int(
                        epoch_rng.integers(method.action_prior.max_length, length + 1)
                    )
                    action_start = action_stop - method.action_prior.max_length
                    # CausalActionPrior applies the same rolling truncation at
                    # evaluation time.  Passing the real prefix reproduces
                    # that path and returns exactly max_length predictions.
                    action_history = actions[: action_stop - 1]
                else:
                    maximum_prefix = min(
                        int(action_cfg.get("history_prefix_max", 0)),
                        method.action_prior.max_length - length,
                    )
                    action_prefix = int(epoch_rng.integers(maximum_prefix + 1))
                    if action_prefix:
                        prefix = torch.as_tensor(
                            epoch_rng.uniform(
                                -1.0,
                                1.0,
                                size=(action_prefix, actions.size(-1)),
                            ),
                            dtype=actions.dtype,
                            device=device,
                        )
                        action_history = torch.cat((prefix, action_history), 0)
                noise_std = float(action_cfg.get("history_noise_std", 0.0))
                dropout = float(action_cfg.get("history_dropout", 0.0))
                if noise_std > 0 or dropout > 0:
                    action_history = action_history.clone()
                    augmented_start = action_prefix
                    if noise_std > 0 and action_history.size(0) > augmented_start:
                        noise = torch.as_tensor(
                            epoch_rng.normal(
                                0.0,
                                noise_std,
                                size=(
                                    action_history.size(0) - augmented_start,
                                    actions.size(-1),
                                ),
                            ),
                            dtype=actions.dtype,
                            device=device,
                        )
                        action_history[augmented_start:] = (
                            action_history[augmented_start:] + noise
                        ).clamp(method.action_prior.low, method.action_prior.high)
                    if dropout > 0 and action_history.size(0) > augmented_start:
                        dropped = torch.as_tensor(
                            epoch_rng.random(action_history.size(0) - augmented_start)
                            < dropout,
                            dtype=torch.bool,
                            device=device,
                        )
                        action_history[augmented_start:][dropped] = 0
            with torch.no_grad():
                if state_enabled:
                    state_sequence = torch.cat(
                        (
                            states[state_start:state_stop],
                            next_states[state_stop - 1 : state_stop],
                        ),
                        0,
                    )
                    mask_sequence = torch.cat(
                        (
                            masks[state_start:state_stop],
                            next_masks[state_stop - 1 : state_stop],
                        ),
                        0,
                    )
                    video_sequence = torch.cat(
                        (
                            video_tokens[state_start:state_stop],
                            next_video_tokens[state_stop - 1 : state_stop],
                        ),
                        0,
                    )
                    state_valid = torch.ones(
                        1,
                        state_window + 1,
                        dtype=torch.bool,
                        device=device,
                    )
                state_output = (
                    method.state_prior(
                        state_sequence.unsqueeze(0),
                        mask_sequence.unsqueeze(0),
                        video_sequence.unsqueeze(0),
                        state_valid,
                    )
                    if state_enabled
                    else None
                )
                action_output = (
                    method.action_prior(action_history.unsqueeze(0))
                    if action_enabled
                    else None
                )
            loss = states.new_zeros(())
            has_trainable_loss = False
            if state_enabled:
                assert state_output is not None
                predicted, _, state_condition = method.state_adapter(
                    states[state_start:state_stop],
                    state_output.passive_next.squeeze(0),
                    actions[state_start:state_stop],
                    state_output.hidden.squeeze(0),
                    state_output.video_summary[:, :-1].squeeze(0),
                )
                state_loss = masked_state_loss(
                    predicted,
                    next_states[state_start:state_stop],
                    next_masks[state_start:state_stop],
                    state_dimension_weights,
                )
                train_horizon = int(config["model"]["state_adapter"].get("train_horizon", 1))
                if train_horizon > 1:
                    max_start = max(0, length - train_horizon)
                    rollout_start = int(epoch_rng.integers(max_start + 1))
                    state_loss = state_loss + float(
                        config["model"]["state_adapter"].get("lambda_multi", 1.0)
                    ) * _multistep_state_loss(
                        method,
                        states,
                        masks,
                        next_states,
                        next_masks,
                        actions,
                        video_tokens,
                        next_video_tokens,
                        train_horizon,
                        float(config["model"]["state_adapter"].get("rollout_discount", 0.99)),
                        start=rollout_start,
                        context_length=int(
                            config["model"]["state_adapter"].get(
                                "train_context_length", 8
                            )
                        ),
                        dimension_weights=state_dimension_weights,
                    )
                adapted_video_logits, _ = method.state_adapter.adapt_video_sequence(
                    state_output.video_logits.squeeze(0),
                    state_output.video_predictor_hidden.squeeze(0),
                    state_condition,
                )
                token_targets = next_video_tokens[state_start:state_stop].flatten(-2)
                video_loss = F.cross_entropy(
                    adapted_video_logits.reshape(-1, adapted_video_logits.size(-1)),
                    token_targets.reshape(-1),
                )
                combined_state_loss = state_loss + float(
                    config["model"]["state_adapter"].get("video_loss_weight", 1.0)
                ) * video_loss
                loss = loss + combined_state_loss
                total_state += float(combined_state_loss.item())
                total_video += float(video_loss.item())
                state_updates += 1
                has_trainable_loss = True
            if action_enabled:
                assert action_output is not None
                output_slice = slice(
                    action_prefix,
                    action_prefix + action_stop - action_start,
                )
                base = TanhNormal(
                    action_output.distribution.loc.squeeze(0)[output_slice],
                    action_output.distribution.log_scale.squeeze(0)[output_slice],
                    action_output.distribution.low,
                    action_output.distribution.high,
                )
                adapted, _, _ = method.action_adapter(
                    base,
                    action_output.hidden.squeeze(0)[output_slice],
                    states[action_start:action_stop],
                    goals[action_start:action_stop],
                )
                selected_expert = action_is_expert[action_start:action_stop]
                action_targets = torch.as_tensor(
                    episode.get("expert_actions", episode["actions"]),
                    device=device,
                )
                selected_actions = action_targets[action_start:action_stop]
                if selected_expert.any():
                    selected_adapted = TanhNormal(
                        adapted.loc[selected_expert],
                        adapted.log_scale[selected_expert],
                        adapted.low,
                        adapted.high,
                    )
                    selected_base = TanhNormal(
                        base.loc[selected_expert],
                        base.log_scale[selected_expert],
                        base.low,
                        base.high,
                    )
                    objective = str(
                        action_cfg.get(
                            "objective", "soft_target_mle_with_conservative_kl"
                        )
                    )
                    if objective == "maximum_likelihood":
                        likelihood_clip = float(
                            action_cfg.get("likelihood_clip", 1e-3)
                        )
                        action_range = selected_adapted.high - selected_adapted.low
                        likelihood_actions = torch.maximum(
                            torch.minimum(
                                selected_actions[selected_expert],
                                selected_adapted.high - likelihood_clip * action_range,
                            ),
                            selected_adapted.low + likelihood_clip * action_range,
                        )
                        action_loss = -selected_adapted.log_prob(
                            likelihood_actions
                        ).mean()
                        soft_nll = action_loss
                        conservative_kl = action_loss.new_zeros(())
                    elif objective == "soft_target_mle_with_conservative_kl":
                        action_loss, soft_nll, conservative_kl = (
                            soft_action_adapter_loss(
                                selected_adapted,
                                selected_base,
                                selected_actions[selected_expert],
                                target_sigma=float(action_cfg.get("target_sigma", 0.1)),
                                target_samples=int(action_cfg.get("target_samples", 4)),
                                label_smoothing=float(action_cfg.get("label_smoothing", 0.02)),
                                likelihood_clip=float(action_cfg.get("likelihood_clip", 1e-3)),
                                kl_weight=kl_weight,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown Action Adapter objective: {objective}"
                        )
                    loss = loss + action_loss
                    total_action += float(action_loss.item())
                    total_soft_nll += float(soft_nll.item())
                    total_action_kl += float(conservative_kl.item())
                    action_updates += 1
                    has_trainable_loss = True
            assert optimizer is not None
            if not has_trainable_loss:
                continue
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite adapter loss in epoch {epoch + 1}, episode {int(episode_index)}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if state_gradient_clip_norm > 0 and state_parameters:
                torch.nn.utils.clip_grad_norm_(
                    state_parameters,
                    state_gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            if action_gradient_clip_norm > 0 and action_parameters:
                torch.nn.utils.clip_grad_norm_(
                    action_parameters,
                    action_gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            optimizer.step()
            steps += 1
        if scheduler is not None:
            scheduler.step()
        state_average = total_state / max(1, state_updates)
        action_average = total_action / max(1, action_updates)
        soft_average = total_soft_nll / max(1, action_updates)
        kl_average = total_action_kl / max(1, action_updates)
        history["state_adapter"].append(state_average)
        history["video_adapter"].append(total_video / max(1, state_updates))
        history["action_adapter"].append(action_average)
        history["action_soft_nll"].append(soft_average)
        history["action_conservative_kl"].append(kl_average)
        history["kl_weight"].append(kl_weight)
        history["learning_rate"].append(
            float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
        )
        print(
            f"epoch={epoch + 1} state_adapter_loss={state_average:.6f} "
            f"video_adapter_loss={total_video / max(1, state_updates):.6f} "
            f"action_adapter_loss={action_average:.6f} soft_nll={soft_average:.6f} "
            f"conservative_kl={kl_average:.6f} kl_weight={kl_weight:.4f}"
        )
    output = Path(
        args.output
        or config.get(
            "output",
            checkpoint_path(config, "adapters", f"{env_name}_adapters.pt"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "adapters",
            "architecture_version": method.state_prior.architecture_version,
            "config": config,
            "state_adapter": method.state_adapter.state_dict() if state_enabled else None,
            "action_adapter": method.action_adapter.state_dict() if action_enabled else None,
            "parameter_report": report,
            "wall_clock_train_seconds": time.perf_counter() - started,
            "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "history": history,
        },
        output,
    )
    diagnostics_directory = output.parent / f"{output.stem}_diagnostics"
    save_training_history(history, diagnostics_directory / "training_loss")
    raw_diagnostic_data = LazyEpisodeDataset(
        data,
        split="val",
        seed=seed,
        max_transitions=config.get("max_transitions"),
    )
    token_diagnostic_data = (
        LazyEpisodeDataset(
            token_root,
            split="val",
            seed=seed,
            max_transitions=config.get("max_transitions"),
        )
        if state_enabled and len(raw_diagnostic_data)
        else None
    )
    diagnostic_data = (
        _StateTokenDataset(raw_diagnostic_data, token_diagnostic_data)
        if token_diagnostic_data is not None
        else dataset
    )
    if len(diagnostic_data) and state_enabled and action_enabled:
        diagnostic_episode, diagnostic_tokens = diagnostic_data[0]
        paired_config = dict(config["env"].get("paired_data", {}))
        if bool(paired_config.get("successful_expert_only", False)):
            threshold = float(paired_config.get("success_reward_threshold", 0.0))
            for candidate, candidate_tokens in diagnostic_data:
                if np.any(np.asarray(candidate["rewards"]) >= threshold):
                    diagnostic_episode = candidate
                    diagnostic_tokens = candidate_tokens
                    break
        metrics = _save_adapter_diagnostics(
            method,
            diagnostic_episode,
            diagnostic_tokens,
            device,
            diagnostics_directory,
            state_enabled,
            action_enabled,
            paired_config,
        )
        print(f"Adapter diagnostics: {json.dumps(metrics, sort_keys=True)}")
    print(f"Saved adapters to {output}")


if __name__ == "__main__":
    main()
