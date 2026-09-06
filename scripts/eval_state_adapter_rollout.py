"""Open-loop diagnostics for the action-conditioned State Adapter.

The evaluator branches from a real observation history and rolls the frozen
State Prior and State Prior + State Adapter forward under the same recorded
actions.  Neither branch sees future simulator observations after branching.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.envs.rendering import render_windy
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.visualization import save_counterfactual

from ._common import config_from_unknown


def _load_state_models(method, state_path: Path, adapter_path: Path) -> None:
    state_checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    expected = int(method.state_prior.architecture_version)
    if int(state_checkpoint.get("architecture_version", 0)) != expected:
        raise ValueError(
            f"State Prior checkpoint {state_path} is incompatible with architecture "
            f"version {expected}"
        )
    state = state_checkpoint.get("state", {})
    if "state_prior" not in state or "tokenizer" not in state:
        raise ValueError(
            f"State Prior checkpoint {state_path} must bundle state_prior and tokenizer"
        )
    method.state_prior.load_state_dict(state["state_prior"])
    method.tokenizer.load_state_dict(state["tokenizer"])

    adapter_checkpoint = torch.load(
        adapter_path, map_location="cpu", weights_only=False
    )
    if int(adapter_checkpoint.get("architecture_version", 0)) != expected:
        raise ValueError(
            f"Adapter checkpoint {adapter_path} is incompatible with architecture "
            f"version {expected}"
        )
    adapter_state = adapter_checkpoint.get("state_adapter")
    if adapter_state is None:
        raise ValueError(f"Checkpoint {adapter_path} contains no trained State Adapter")
    method.state_adapter.load_state_dict(adapter_state)
    method.use_state_adapter = True
    method.freeze_priors()
    method.eval()


@torch.inference_mode()
def _episode_tokens(
    method,
    episode: dict[str, np.ndarray],
    token_episode: dict[str, np.ndarray | torch.Tensor] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_episode is not None:
        if "next_video_tokens" not in token_episode:
            raise ValueError("Paired token cache is missing next_video_tokens")
        return (
            torch.as_tensor(token_episode["video_tokens"], device=device).long(),
            torch.as_tensor(token_episode["next_video_tokens"], device=device).long(),
        )

    def encode(frames: np.ndarray, batch_size: int = 128) -> torch.Tensor:
        chunks = []
        for start in range(0, len(frames), int(batch_size)):
            chunks.append(
                method.tokenizer.encode(
                    torch.as_tensor(
                        frames[start : start + int(batch_size)], device=device
                    )
                )
            )
        return torch.cat(chunks, dim=0)

    return (
        encode(episode["rgb"]),
        encode(episode["next_rgb"]),
    )


@torch.inference_mode()
def _predict_next(
    method,
    state_context: torch.Tensor,
    mask_context: torch.Tensor,
    token_context: torch.Tensor,
    action: torch.Tensor,
    *,
    adapted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.ones(
        state_context.size(0),
        state_context.size(1),
        dtype=torch.bool,
        device=state_context.device,
    )
    prior = method.state_prior.encode_context(
        state_context, mask_context, token_context, valid
    )
    logit_adapter = None
    if adapted:
        next_state, _, condition = method.state_adapter(
            state_context[:, -1],
            prior.passive_next,
            action,
            prior.observation_hidden,
            prior.video_summary,
        )

        def logit_adapter(logits, token_hidden, spatial_index):
            return method.state_adapter.adapt_video_logits(
                logits, token_hidden, spatial_index, condition
            )[0]

    else:
        next_state = prior.passive_next
    next_tokens, _, _ = method.state_prior.generate_next_video(
        prior.video_cache,
        prior.observation_hidden,
        logit_adapter=logit_adapter,
    )
    return next_state, next_tokens.reshape(
        next_tokens.size(0), *token_context.shape[-2:]
    )


def _append_context(
    context: torch.Tensor, value: torch.Tensor, maximum: int
) -> torch.Tensor:
    return torch.cat((context, value[:, None]), dim=1)[:, -int(maximum) :]


@torch.inference_mode()
def rollout_from_history(
    method,
    episode: dict[str, np.ndarray],
    token_episode: dict[str, np.ndarray | torch.Tensor] | None,
    history_steps: int,
    max_rollout_steps: int,
    device: torch.device,
    *,
    expert_only: bool = False,
) -> dict[str, Any]:
    """Compare passive and adapted open-loop futures from one real prefix."""

    length = min(
        int(episode["sequence_length"]),
        len(episode["actions"]),
        len(episode["next_control_state"]),
    )
    history_steps = int(history_steps)
    if history_steps < 0 or history_steps >= length:
        raise ValueError(
            f"history_steps={history_steps} must be in [0, {length - 1}]"
        )
    rollout_steps = length - history_steps
    if int(max_rollout_steps) > 0:
        rollout_steps = min(rollout_steps, int(max_rollout_steps))
    if expert_only:
        labels = np.asarray(
            episode.get("action_is_expert", np.zeros(length, bool)), bool
        )[history_steps : history_steps + rollout_steps]
        first_nonexpert = np.flatnonzero(~labels)
        if first_nonexpert.size:
            rollout_steps = int(first_nonexpert[0])
    if rollout_steps <= 0:
        raise ValueError("No eligible rollout actions after the selected history")

    states = torch.as_tensor(episode["control_state"][:length], device=device)
    masks = torch.as_tensor(
        episode["state_mask"][:length], dtype=torch.bool, device=device
    )
    next_states = torch.as_tensor(
        episode["next_control_state"][:length], device=device
    )
    next_masks = torch.as_tensor(
        episode["next_state_mask"][:length], dtype=torch.bool, device=device
    )
    actions = torch.as_tensor(episode["actions"][:length], device=device)
    current_tokens, next_tokens = _episode_tokens(
        method, episode, token_episode, device
    )
    context_limit = int(method.state_prior.context_frames)
    prefix_start = max(0, history_steps + 1 - context_limit)
    state_prefix = states[prefix_start : history_steps + 1].unsqueeze(0)
    mask_prefix = masks[prefix_start : history_steps + 1].unsqueeze(0)
    token_prefix = current_tokens[prefix_start : history_steps + 1].unsqueeze(0)

    branches = {
        "passive": [state_prefix.clone(), mask_prefix.clone(), token_prefix.clone()],
        "adapted": [state_prefix.clone(), mask_prefix.clone(), token_prefix.clone()],
    }
    predicted_states = {
        name: [states[history_steps].detach().cpu()] for name in branches
    }
    predicted_tokens = {
        name: [current_tokens[history_steps].detach().cpu()] for name in branches
    }
    for offset in range(rollout_steps):
        action = actions[history_steps + offset : history_steps + offset + 1]
        for name, (state_context, mask_context, token_context) in branches.items():
            state, tokens = _predict_next(
                method,
                state_context,
                mask_context,
                token_context,
                action,
                adapted=name == "adapted",
            )
            if not torch.isfinite(state).all():
                raise FloatingPointError(
                    f"Non-finite {name} rollout at history={history_steps}, "
                    f"horizon={offset + 1}"
                )
            # State masks describe the task schema and are constant during an
            # episode. Reusing the latest known mask avoids future-state leakage.
            predicted_mask = mask_context[:, -1]
            branches[name] = [
                _append_context(state_context, state, context_limit),
                _append_context(mask_context, predicted_mask, context_limit),
                _append_context(token_context, tokens, context_limit),
            ]
            predicted_states[name].append(state.squeeze(0).float().cpu())
            predicted_tokens[name].append(tokens.squeeze(0).long().cpu())

    stop = history_steps + rollout_steps
    true_states = torch.cat(
        (states[history_steps : history_steps + 1], next_states[history_steps:stop]),
        dim=0,
    ).float().cpu()
    true_masks = torch.cat(
        (masks[history_steps : history_steps + 1], next_masks[history_steps:stop]),
        dim=0,
    ).bool().cpu()
    true_tokens = torch.cat(
        (
            current_tokens[history_steps : history_steps + 1],
            next_tokens[history_steps:stop],
        ),
        dim=0,
    ).long().cpu()
    true_frames = np.concatenate(
        (
            np.asarray(episode["rgb"][history_steps : history_steps + 1], np.uint8),
            np.asarray(episode["next_rgb"][history_steps:stop], np.uint8),
        ),
        axis=0,
    )
    goals = episode.get("goals")
    goal = (
        np.asarray(goals[history_steps], np.float32)
        if goals is not None and len(goals) > history_steps
        else np.empty(0, np.float32)
    )
    return {
        "history_steps": history_steps,
        "rollout_steps": rollout_steps,
        "actions": actions[history_steps:stop].float().cpu().numpy(),
        "true_states": true_states.numpy(),
        "true_masks": true_masks.numpy(),
        "true_tokens": true_tokens.numpy(),
        "true_frames": true_frames,
        "observed_states": states[: history_steps + 1].float().cpu().numpy(),
        "goal": goal,
        "passive_states": torch.stack(predicted_states["passive"]).numpy(),
        "adapted_states": torch.stack(predicted_states["adapted"]).numpy(),
        "passive_tokens": torch.stack(predicted_tokens["passive"]).numpy(),
        "adapted_tokens": torch.stack(predicted_tokens["adapted"]).numpy(),
    }


def rollout_curves(result: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    mask = np.asarray(result["true_masks"], bool)
    true = np.asarray(result["true_states"], np.float32)

    def state_errors(name: str) -> tuple[np.ndarray, np.ndarray]:
        difference = np.asarray(result[f"{name}_states"], np.float32) - true
        squared = np.where(mask, np.square(difference), 0.0)
        count = np.maximum(mask.sum(axis=-1), 1)
        return np.sqrt(squared.sum(axis=-1) / count), np.sqrt(squared.sum(axis=-1))

    passive_rmse, passive_l2 = state_errors("passive")
    adapted_rmse, adapted_l2 = state_errors("adapted")
    true_tokens = np.asarray(result["true_tokens"])
    curves = {
        "passive_state_rmse": passive_rmse,
        "adapted_state_rmse": adapted_rmse,
        "passive_state_l2": passive_l2,
        "adapted_state_l2": adapted_l2,
        "passive_video_accuracy": np.mean(
            np.asarray(result["passive_tokens"]) == true_tokens, axis=(-2, -1)
        ),
        "adapted_video_accuracy": np.mean(
            np.asarray(result["adapted_tokens"]) == true_tokens, axis=(-2, -1)
        ),
    }
    env = config["env"]
    if env["name"] == "windy":
        for name in ("passive", "adapted"):
            difference = np.asarray(result[f"{name}_states"]) - true
            curves[f"{name}_position_l2"] = np.linalg.norm(
                difference[:, :2], axis=-1
            )
            curves[f"{name}_velocity_l2"] = np.linalg.norm(
                difference[:, 2:4], axis=-1
            )
    achieved_slice = (
        None
        if env["name"] == "windy"
        else env.get(
            "achieved_goal_slice", env.get("action_adapter_achieved_goal_slice")
        )
    )
    if achieved_slice is not None:
        start, stop = map(int, achieved_slice)
        for name in ("passive", "adapted"):
            difference = np.asarray(result[f"{name}_states"]) - true
            curves[f"{name}_achieved_goal_l2"] = np.linalg.norm(
                difference[:, start:stop], axis=-1
            )
    return curves


def metric_row(
    result: dict[str, Any], curves: dict[str, np.ndarray], episode_id: int
) -> dict[str, float | int]:
    forecast = slice(1, None)
    passive = curves["passive_state_rmse"][forecast]
    adapted = curves["adapted_state_rmse"][forecast]
    passive_mean = float(np.mean(passive))
    adapted_mean = float(np.mean(adapted))
    row: dict[str, float | int] = {
        "episode_id": int(episode_id),
        "history_steps": int(result["history_steps"]),
        "rollout_steps": int(result["rollout_steps"]),
        "passive_state_rmse": passive_mean,
        "adapted_state_rmse": adapted_mean,
        "state_rmse_improvement_fraction": float(
            (passive_mean - adapted_mean) / max(passive_mean, 1e-12)
        ),
        "passive_final_state_rmse": float(curves["passive_state_rmse"][-1]),
        "adapted_final_state_rmse": float(curves["adapted_state_rmse"][-1]),
        "passive_video_token_accuracy": float(
            np.mean(curves["passive_video_accuracy"][forecast])
        ),
        "adapted_video_token_accuracy": float(
            np.mean(curves["adapted_video_accuracy"][forecast])
        ),
    }
    for horizon in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if horizon > int(result["rollout_steps"]):
            continue
        passive_h = float(curves["passive_state_rmse"][horizon])
        adapted_h = float(curves["adapted_state_rmse"][horizon])
        row[f"passive_state_rmse_h{horizon}"] = passive_h
        row[f"adapted_state_rmse_h{horizon}"] = adapted_h
        row[f"state_rmse_improvement_fraction_h{horizon}"] = float(
            (passive_h - adapted_h) / max(passive_h, 1e-12)
        )
        row[f"passive_video_token_accuracy_h{horizon}"] = float(
            curves["passive_video_accuracy"][horizon]
        )
        row[f"adapted_video_token_accuracy_h{horizon}"] = float(
            curves["adapted_video_accuracy"][horizon]
        )
    for suffix in ("position_l2", "velocity_l2", "achieved_goal_l2"):
        for branch in ("passive", "adapted"):
            key = f"{branch}_{suffix}"
            if key in curves:
                row[f"{key}_mean"] = float(np.mean(curves[key][forecast]))
                row[f"{key}_final"] = float(curves[key][-1])
    return row


@torch.inference_mode()
def _decoded_frames(method, tokens: np.ndarray, device: torch.device) -> list[np.ndarray]:
    tensor = torch.as_tensor(tokens, device=device).long()
    decoded = method.tokenizer.decode(tensor).clamp(0, 1)
    frames = (
        decoded.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    return list(frames)


def _trajectory_state_slice(config: dict[str, Any]) -> tuple[slice, str]:
    """Choose the task-space coordinates that best describe the rollout."""

    env = config["env"]
    if env["name"] == "windy":
        return slice(0, 2), "agent position"
    achieved = env.get(
        "achieved_goal_slice", env.get("action_adapter_achieved_goal_slice")
    )
    if achieved is not None and int(achieved[1]) - int(achieved[0]) >= 2:
        return slice(int(achieved[0]), int(achieved[0]) + 2), "achieved-goal position"
    return slice(0, 2), "first two control-state coordinates"


def _trajectory_coordinates(
    states: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    state_slice, _ = _trajectory_state_slice(config)
    return np.asarray(states, np.float32)[..., state_slice]


def _semantic_state_frames(
    result: dict[str, Any], config: dict[str, Any], branch: str
) -> list[np.ndarray]:
    """Render a moving task-space marker when simulator state is not invertible.

    Fetch and Humanoid datasets intentionally store model observations rather
    than private MuJoCo qpos/qvel.  Those observations cannot reconstruct a
    physically exact simulator frame.  A task-space trajectory canvas is an
    honest state rendering and avoids pretending that VQ token reconstruction
    is the State Adapter's predicted state.
    """

    predicted = _trajectory_coordinates(result[f"{branch}_states"], config)
    observed = _trajectory_coordinates(result["observed_states"], config)
    true = _trajectory_coordinates(result["true_states"], config)
    coordinate_sets = [predicted, observed, true]
    goal = np.asarray(result.get("goal", np.empty(0)), np.float32)
    if goal.size >= 2:
        coordinate_sets.append(goal[:2][None])
    finite = np.concatenate(coordinate_sets, axis=0)
    finite = finite[np.isfinite(finite).all(axis=-1)]
    if not len(finite):
        finite = np.asarray([[-1.0, -1.0], [1.0, 1.0]], np.float32)
    lower, upper = finite.min(axis=0), finite.max(axis=0)
    span = np.maximum(upper - lower, 1e-3)
    lower -= 0.12 * span
    upper += 0.12 * span
    width = int(result["true_frames"].shape[2])
    height = int(result["true_frames"].shape[1])
    margin = max(5, min(width, height) // 12)

    def point(value: np.ndarray) -> tuple[int, int]:
        normalized = (value - lower) / np.maximum(upper - lower, 1e-6)
        x = margin + normalized[0] * max(width - 2 * margin - 1, 1)
        y = height - margin - normalized[1] * max(height - 2 * margin - 1, 1)
        return int(round(x)), int(round(y))

    color = (190, 70, 55) if branch == "passive" else (45, 105, 180)
    frames: list[np.ndarray] = []
    for step in range(len(predicted)):
        image = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (margin, margin, width - margin, height - margin),
            outline=(205, 205, 205),
        )
        observed_points = [point(value) for value in observed]
        if len(observed_points) > 1:
            draw.line(observed_points, fill=(145, 145, 145), width=2)
        predicted_points = [point(value) for value in predicted[: step + 1]]
        if len(predicted_points) > 1:
            draw.line(predicted_points, fill=color, width=2)
        if goal.size >= 2 and np.isfinite(goal[:2]).all():
            gx, gy = point(goal[:2])
            draw.line((gx - 4, gy, gx + 4, gy), fill=(40, 135, 75), width=2)
            draw.line((gx, gy - 4, gx, gy + 4), fill=(40, 135, 75), width=2)
        x, y = predicted_points[-1]
        radius = max(3, min(width, height) // 24)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        frames.append(np.asarray(image, np.uint8))
    return frames


def _state_rendered_frames(
    result: dict[str, Any], config: dict[str, Any], branch: str
) -> list[np.ndarray]:
    """Render the predicted control state rather than decoded video tokens."""

    if config["env"]["name"] != "windy":
        return _semantic_state_frames(result, config, branch)
    env = config["env"]
    size = int(env.get("image_size", [64, 64])[0])
    goal = np.asarray(result.get("goal", np.zeros(2)), np.float32)
    if goal.size < 2:
        goal = np.zeros(2, np.float32)
    colors = env.get(
        "region_colors",
        [[60, 70, 80], [70, 80, 70], [80, 70, 60], [70, 60, 80]],
    )
    frames = []
    for state in np.asarray(result[f"{branch}_states"], np.float32):
        # Long open-loop predictions may leave the physical [-1, 1] world.
        # Keep the marker visible at the boundary; the PNG/NPZ retain the exact
        # unclipped trajectory and therefore still expose the divergence.
        position = np.clip(state[:2], -1.0, 1.0)
        frames.append(render_windy(size, position, state[2:4], goal[:2], colors))
    return frames


def save_trajectory_plot(
    result: dict[str, Any],
    curves: dict[str, np.ndarray],
    config: dict[str, Any],
    output: Path,
) -> None:
    """Plot the real future directly against both open-loop predictions."""

    observed = _trajectory_coordinates(result["observed_states"], config)
    true = _trajectory_coordinates(result["true_states"], config)
    passive = _trajectory_coordinates(result["passive_states"], config)
    adapted = _trajectory_coordinates(result["adapted_states"], config)
    _, coordinate_name = _trajectory_state_slice(config)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        observed[:, 0],
        observed[:, 1],
        color="0.6",
        linewidth=2,
        label="real history",
    )
    axes[0].plot(
        true[:, 0],
        true[:, 1],
        color="#238b45",
        linewidth=2,
        label="real future",
    )
    axes[0].plot(
        passive[:, 0],
        passive[:, 1],
        "--",
        color="#cb3d32",
        linewidth=1.8,
        label="State Prior",
    )
    axes[0].plot(
        adapted[:, 0],
        adapted[:, 1],
        "--",
        color="#2468a2",
        linewidth=1.8,
        label="+ State Adapter",
    )
    axes[0].scatter(
        true[0, 0],
        true[0, 1],
        color="#111111",
        s=28,
        zorder=4,
        label="rollout start",
    )
    goal = np.asarray(result.get("goal", np.empty(0)), np.float32)
    if goal.size >= 2 and np.isfinite(goal[:2]).all():
        axes[0].scatter(
            goal[0],
            goal[1],
            marker="x",
            color="#111111",
            s=55,
            zorder=4,
            label="goal",
        )
    if config["env"]["name"] == "windy":
        axes[0].add_patch(
            plt.Rectangle(
                (-1.0, -1.0),
                2.0,
                2.0,
                fill=False,
                edgecolor="0.45",
                linestyle=":",
                linewidth=1.2,
                label="physical world boundary",
            )
        )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(
        xlabel=f"{coordinate_name} x",
        ylabel=f"{coordinate_name} y",
        title="Real and predicted trajectories",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")

    horizon = np.arange(int(result["rollout_steps"]) + 1)
    axes[1].plot(
        horizon,
        curves["passive_state_rmse"],
        color="#cb3d32",
        label="State Prior state RMSE",
    )
    axes[1].plot(
        horizon,
        curves["adapted_state_rmse"],
        color="#2468a2",
        label="+ State Adapter state RMSE",
    )
    for key, label, color in (
        ("passive_position_l2", "State Prior position L2", "#ee8a72"),
        ("adapted_position_l2", "+ Adapter position L2", "#6f9fd1"),
        ("passive_achieved_goal_l2", "State Prior task-position L2", "#ee8a72"),
        ("adapted_achieved_goal_l2", "+ Adapter task-position L2", "#6f9fd1"),
    ):
        if key in curves:
            axes[1].plot(
                horizon,
                curves[key],
                linestyle=":",
                color=color,
                label=label,
            )
    axes[1].set(
        xlabel="open-loop horizon",
        ylabel="error to real trajectory",
        title="Open-loop error growth",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        f"Full remaining rollout from history={result['history_steps']} "
        f"({result['rollout_steps']} predicted steps)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_rollout_artifacts(
    method,
    result: dict[str, Any],
    curves: dict[str, np.ndarray],
    config: dict[str, Any],
    output: Path,
    stem: str,
    device: torch.device,
    gif_scale: int,
) -> None:
    np.savez_compressed(
        output / f"{stem}.npz",
        **{
            key: value
            for key, value in result.items()
            if isinstance(value, (np.ndarray, int, float))
        },
        **curves,
    )
    passive_frames = _state_rendered_frames(result, config, "passive")
    adapted_frames = _state_rendered_frames(result, config, "adapted")
    save_counterfactual(
        passive_frames,
        adapted_frames,
        list(result["true_frames"]),
        output / f"{stem}.gif",
        scale=gif_scale,
    )

    # Keep video-token generation visible as a separate diagnostic.  It must
    # not be confused with rendering the State Adapter's state prediction.
    decoded_passive = [
        result["true_frames"][0],
        *_decoded_frames(method, result["passive_tokens"][1:], device),
    ]
    decoded_adapted = [
        result["true_frames"][0],
        *_decoded_frames(method, result["adapted_tokens"][1:], device),
    ]
    save_counterfactual(
        decoded_passive,
        decoded_adapted,
        list(result["true_frames"]),
        output / f"{stem}_video_tokens.gif",
        scale=gif_scale,
    )

    save_trajectory_plot(result, curves, config, output / f"{stem}.png")
    horizon = np.arange(int(result["rollout_steps"]) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(horizon, curves["passive_video_accuracy"], label="State Prior")
    axis.plot(horizon, curves["adapted_video_accuracy"], label="+ State Adapter")
    axis.set(
        xlabel="open-loop horizon",
        ylabel="video-token accuracy",
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.suptitle(f"Video-token diagnostic, history={result['history_steps']}")
    figure.tight_layout()
    figure.savefig(output / f"{stem}_video_tokens.png", dpi=170)
    plt.close(figure)


def _aggregate(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    by_history: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        by_history[int(row["history_steps"])].append(row)

    def summarize(selected: list[dict[str, float | int]]) -> dict[str, float]:
        numeric = {
            key
            for row in selected
            for key, value in row.items()
            if key not in {"episode_id", "history_steps"}
            and isinstance(value, (int, float, np.number))
        }
        return {
            key: float(np.mean([float(row[key]) for row in selected if key in row]))
            for key in sorted(numeric)
        }

    return {
        "rollouts": len(rows),
        "overall": summarize(rows),
        "by_history": {
            str(history): {"rollouts": len(selected), **summarize(selected)}
            for history, selected in sorted(by_history.items())
        },
    }


def save_summary_plot(rows: list[dict[str, float | int]], output: Path) -> None:
    summary = _aggregate(rows)["by_history"]
    histories = sorted(map(int, summary))
    passive = [summary[str(value)]["passive_state_rmse"] for value in histories]
    adapted = [summary[str(value)]["adapted_state_rmse"] for value in histories]
    passive_video = [
        summary[str(value)]["passive_video_token_accuracy"] for value in histories
    ]
    adapted_video = [
        summary[str(value)]["adapted_video_token_accuracy"] for value in histories
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(histories, passive, "o-", label="State Prior")
    axes[0].plot(histories, adapted, "o-", label="+ State Adapter")
    axes[0].set(xlabel="real history steps", ylabel="mean open-loop state RMSE")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(histories, passive_video, "o-", label="State Prior")
    axes[1].plot(histories, adapted_video, "o-", label="+ State Adapter")
    axes[1].set(xlabel="real history steps", ylabel="mean video-token accuracy", ylim=(-0.02, 1.02))
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate State Adapter open-loop observation/video correction"
    )
    parser.add_argument("--data", required=True, help="Held-out paired dataset")
    parser.add_argument(
        "--tokens",
        help="Optional paired token cache; raw RGB is encoded when omitted",
    )
    parser.add_argument("--state-prior", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--history-lengths", nargs="+", type=int, default=[0, 1, 4, 8])
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=0,
        help="Maximum predicted steps; 0 (default) rolls to the end of the episode",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-visualizations", type=int, default=4)
    parser.add_argument(
        "--gif-scale",
        type=int,
        default=0,
        help="Nearest-neighbor display scale; 0 selects 4x for Windy and 2x otherwise",
    )
    parser.add_argument(
        "--expert-only",
        action="store_true",
        help="Evaluate only contiguous actions labelled as expert",
    )
    args, unknown = parser.parse_known_args()
    if args.max_rollout_steps < 0 or args.episodes <= 0:
        parser.error("rollout steps must be non-negative and episodes must be positive")
    if args.max_visualizations < 0:
        parser.error("--max-visualizations must be non-negative")
    if args.gif_scale < 0:
        parser.error("--gif-scale must be non-negative")
    histories = sorted(set(args.history_lengths))
    if not histories or histories[0] < 0:
        parser.error("--history-lengths must contain non-negative integers")

    config = config_from_unknown(["model=prior_adapter", *unknown])
    seed = int(config.get("seed", 0))
    raw = LazyEpisodeDataset(args.data, split=args.split, seed=seed)
    if not len(raw):
        raise ValueError(f"No {args.split} episodes in {args.data}")
    tokens = (
        LazyEpisodeDataset(args.tokens, split=args.split, seed=seed)
        if args.tokens
        else None
    )
    if tokens is not None:
        raw_ids = [int(entry["id"]) for entry in raw.entries]
        token_ids = [int(entry["id"]) for entry in tokens.entries]
        if raw_ids != token_ids:
            raise ValueError("Raw and token-cache episode IDs differ")

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    method = build_method(config).to(device)
    _load_state_models(
        method, Path(args.state_prior), Path(args.adapters)
    )
    method.to(device).eval()
    gif_scale = int(args.gif_scale) or (
        4 if config["env"]["name"] == "windy" else 2
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    visualizations = 0
    count = min(len(raw), int(args.episodes))
    progress = tqdm(range(count), desc="state-adapter rollout", unit="episode")
    for index in progress:
        episode = raw[index]
        token_episode = tokens[index] if tokens is not None else None
        if token_episode is None:
            current_tokens, next_tokens = _episode_tokens(
                method, episode, None, device
            )
            # Reuse one bounded-batch RGB encoding for every requested history
            # from this episode.
            token_episode = {
                "video_tokens": current_tokens,
                "next_video_tokens": next_tokens,
            }
        episode_id = int(raw.entries[index]["id"])
        for history in histories:
            if history >= int(episode["sequence_length"]):
                continue
            try:
                result = rollout_from_history(
                    method,
                    episode,
                    token_episode,
                    history,
                    args.max_rollout_steps,
                    device,
                    expert_only=args.expert_only,
                )
            except ValueError as error:
                if "No eligible rollout actions" in str(error):
                    continue
                raise
            curves = rollout_curves(result, config)
            row = metric_row(result, curves, episode_id)
            rows.append(row)
            if visualizations < int(args.max_visualizations):
                stem = f"episode_{episode_id:06d}_history_{history:03d}"
                save_rollout_artifacts(
                    method,
                    result,
                    curves,
                    config,
                    output,
                    stem,
                    device,
                    gif_scale,
                )
                visualizations += 1
        if rows:
            progress.set_postfix(
                adapted_rmse=f"{float(rows[-1]['adapted_state_rmse']):.4f}",
                refresh=False,
            )
    if not rows:
        qualifier = " expert-labelled" if args.expert_only else ""
        raise ValueError(f"No eligible{qualifier} rollout windows were found")

    with (output / "adapter_rollout_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "environment": config["env"]["name"],
        "split": args.split,
        "episodes_requested": int(args.episodes),
        "episodes_loaded": count,
        "history_lengths": histories,
        "max_rollout_steps": int(args.max_rollout_steps),
        "expert_only": bool(args.expert_only),
        "state_prior": str(Path(args.state_prior).resolve()),
        "adapters": str(Path(args.adapters).resolve()),
        **_aggregate(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_summary_plot(rows, output / "adapter_rollout_summary.png")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Saved State Adapter rollout diagnostics to {output}")


if __name__ == "__main__":
    main()
