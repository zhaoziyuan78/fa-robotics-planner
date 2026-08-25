"""Deterministic task scores over predicted state only."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def final_goal_distance(
    states: torch.Tensor,
    goal: torch.Tensor,
    actions: torch.Tensor,
    achieved_slice: slice = slice(0, 2),
    control_cost: float = 0.0,
) -> torch.Tensor:
    achieved = states[:, -1, achieved_slice]
    goal = goal.reshape(1, -1).expand(achieved.size(0), -1)
    return -torch.linalg.vector_norm(achieved - goal, dim=-1) - control_cost * actions.square().sum((1, 2))


def fetch_score(
    task: str,
    states: torch.Tensor,
    goal: torch.Tensor,
    actions: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    goal_dim = goal.numel()
    achieved_bounds = config.get("achieved_goal_slice")
    if achieved_bounds is None:
        achieved = states[:, -1, -goal_dim:]
    else:
        achieved = states[
            :, -1, int(achieved_bounds[0]) : int(achieved_bounds[1])
        ]
    goal_batch = goal.reshape(1, -1)
    goal_dimensions = int(config.get("goal_score_dimensions", goal_dim))
    goal_error = torch.linalg.vector_norm(
        achieved[:, :goal_dimensions] - goal_batch[:, :goal_dimensions], dim=-1
    )
    score = -goal_error
    score -= float(config.get("control_cost", 0.0)) * actions.square().sum((1, 2))
    gripper_slice = config.get("gripper_position_slice", [0, 3])
    gripper = states[
        :, -1, int(gripper_slice[0]) : int(gripper_slice[1])
    ]
    goal_direction = goal_batch[:, :2] - achieved[:, :2]
    goal_direction = goal_direction / torch.linalg.vector_norm(
        goal_direction, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    approach_target = achieved.clone()
    approach_target[:, :2] -= float(config.get("approach_offset", 0.0)) * goal_direction
    xy_error = torch.linalg.vector_norm(
        gripper[:, :2] - approach_target[:, :2], dim=-1
    )
    safe_height = float(config.get("approach_safe_height", 0.0))
    contact_height = float(config.get("approach_contact_height", 0.008))
    xy_threshold = float(config.get("approach_xy_threshold", 0.04))
    approach_target[:, 2] = achieved[:, 2] + torch.where(
        xy_error > xy_threshold,
        torch.full_like(xy_error, safe_height),
        torch.full_like(xy_error, contact_height),
    )
    approach_distance = torch.linalg.vector_norm(gripper - approach_target, dim=-1)
    alignment_reward = float(config.get("contact_alignment_reward", 0.0))
    if alignment_reward:
        contact = approach_distance < float(
            config.get("contact_alignment_distance", 0.06)
        )
        aligned_action = (
            actions[:, :, :2] * goal_direction[:, None, :]
        ).sum(-1).mean(-1).clamp_min(0.0)
        score += alignment_reward * aligned_action * contact.to(score.dtype)
    if task == "fetch_slide":
        velocity_slice = config.get("object_velocity_slice", [-goal_dim - 2, -goal_dim])
        velocity = states[:, -1, int(velocity_slice[0]) : int(velocity_slice[1])]
        terminal_region = goal_error < float(
            config.get("terminal_velocity_distance", float("inf"))
        )
        score -= (
            float(config.get("terminal_velocity_cost", 0.1))
            * torch.linalg.vector_norm(velocity, dim=-1)
            * terminal_region.to(score.dtype)
        )
        moving_toward_goal = (velocity[:, :2] * goal_direction).sum(-1) > float(
            config.get("approach_release_velocity", 0.02)
        )
        score -= float(config.get("approach_cost", 0.0)) * approach_distance * (
            ~moving_toward_goal
        ).to(score.dtype)
    elif task == "fetch_push":
        score -= float(config.get("approach_cost", 0.05)) * approach_distance
    return score


def _np_field(state: np.ndarray, config: Mapping[str, Any], name: str) -> np.ndarray:
    indices = np.asarray(config.get("indices", {}).get(name, []), np.int64)
    if indices.size:
        return state[indices]
    bounds = config.get("slices", {}).get(name)
    if bounds is not None:
        return state[int(bounds[0]) : int(bounds[1])]
    return np.empty(0, np.float32)


def humanoid_score(
    task: str,
    predicted_state: np.ndarray,
    goal: np.ndarray,
    action_sequence: np.ndarray | None,
    config: Mapping[str, Any],
) -> float:
    state = np.asarray(predicted_state, np.float32).reshape(-1)
    weights = config.get("weights", {})
    upright_field = _np_field(state, config, "upright")
    if upright_field.size:
        upright = float(upright_field.mean())
    elif state.size >= 7:
        quaternion = state[3:7]
        upright = float(1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2))
    else:
        upright = 0.0
    height = float(_np_field(state, config, "torso_height").mean()) if _np_field(state, config, "torso_height").size else 0.0
    fall_field = _np_field(state, config, "fall")
    fall = (
        float(fall_field.max())
        if fall_field.size
        else float(height < float(config.get("fall_height", 0.2)))
    )
    score = float(weights.get("upright", 1.0)) * upright - float(weights.get("fall", 10.0)) * fall
    if task == "humanoid_stand":
        score += float(weights.get("height", 1.0)) * height
        velocity = _np_field(state, config, "joint_velocity")
        score -= float(weights.get("velocity", 0.01)) * float(np.square(velocity).sum())
    elif task == "humanoid_balance":
        stability = _np_field(state, config, "root_velocity")
        score -= float(weights.get("stability", 1.0)) * float(np.linalg.norm(stability))
    elif task == "humanoid_reach":
        hand = _np_field(state, config, "hand_position")
        score -= float(weights.get("goal", 1.0)) * float(np.linalg.norm(hand - np.asarray(goal)))
    elif task == "humanoid_push":
        obj, hand = _np_field(state, config, "object_position"), _np_field(state, config, "hand_position")
        score -= float(weights.get("goal", 1.0)) * float(np.linalg.norm(obj - np.asarray(goal)))
        score -= float(weights.get("approach", 0.1)) * float(np.linalg.norm(hand - obj))
    if action_sequence is not None:
        score -= float(weights.get("control", 0.001)) * float(np.square(action_sequence).sum())
    return score


def _torch_field(
    state: torch.Tensor, config: Mapping[str, Any], name: str
) -> torch.Tensor:
    indices = config.get("indices", {}).get(name, [])
    if indices:
        index = torch.as_tensor(indices, dtype=torch.long, device=state.device)
        return state.index_select(-1, index)
    bounds = config.get("slices", {}).get(name)
    if bounds is not None:
        return state[..., int(bounds[0]) : int(bounds[1])]
    return state.new_empty((*state.shape[:-1], 0))


def humanoid_score_torch(
    task: str,
    predicted_states: torch.Tensor,
    goal: torch.Tensor,
    action_sequences: torch.Tensor | None,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Vectorized equivalent of :func:`humanoid_score` for GPU planning."""

    state = predicted_states.reshape(-1, predicted_states.shape[-1])
    weights = config.get("weights", {})
    upright_field = _torch_field(state, config, "upright")
    if upright_field.shape[-1]:
        upright = upright_field.mean(-1)
    elif state.shape[-1] >= 7:
        quaternion = state[:, 3:7]
        upright = 1.0 - 2.0 * (quaternion[:, 1].square() + quaternion[:, 2].square())
    else:
        upright = state.new_zeros(state.shape[0])
    height_field = _torch_field(state, config, "torso_height")
    height = (
        height_field.mean(-1)
        if height_field.shape[-1]
        else state.new_zeros(state.shape[0])
    )
    fall_field = _torch_field(state, config, "fall")
    fall = (
        fall_field.max(-1).values
        if fall_field.shape[-1]
        else (height < float(config.get("fall_height", 0.2))).to(state.dtype)
    )
    score = float(weights.get("upright", 1.0)) * upright
    score = score - float(weights.get("fall", 10.0)) * fall
    if task == "humanoid_stand":
        velocity = _torch_field(state, config, "joint_velocity")
        score = score + float(weights.get("height", 1.0)) * height
        score = score - float(weights.get("velocity", 0.01)) * velocity.square().sum(-1)
    elif task == "humanoid_balance":
        stability = _torch_field(state, config, "root_velocity")
        score = score - float(weights.get("stability", 1.0)) * torch.linalg.vector_norm(
            stability, dim=-1
        )
    elif task == "humanoid_reach":
        hand = _torch_field(state, config, "hand_position")
        score = score - float(weights.get("goal", 1.0)) * torch.linalg.vector_norm(
            hand - goal.reshape(1, -1), dim=-1
        )
    elif task == "humanoid_push":
        object_position = _torch_field(state, config, "object_position")
        hand = _torch_field(state, config, "hand_position")
        score = score - float(weights.get("goal", 1.0)) * torch.linalg.vector_norm(
            object_position - goal.reshape(1, -1), dim=-1
        )
        score = score - float(weights.get("approach", 0.1)) * torch.linalg.vector_norm(
            hand - object_position, dim=-1
        )
    if action_sequences is not None:
        score = score - float(weights.get("control", 0.001)) * action_sequences.square().sum(
            (1, 2)
        )
    return score
