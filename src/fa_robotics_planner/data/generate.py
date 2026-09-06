"""Environment-neutral generators for the three isolated dataset types."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

from fa_robotics_planner.envs.unified import UnifiedControlEnv

from .schemas import DatasetKind
from .writer import EpisodeWriter


def _stack(values: list[np.ndarray], dtype=None) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def ou_actions(
    rng: np.random.Generator,
    length: int,
    low: np.ndarray,
    high: np.ndarray,
    theta: float = 0.15,
    sigma: float = 0.25,
) -> np.ndarray:
    action = np.zeros_like(low, dtype=np.float32)
    result = []
    for _ in range(length):
        action += theta * (-action) + sigma * rng.normal(size=action.shape)
        action = np.clip(action, low, high).astype(np.float32)
        result.append(action.copy())
    return np.stack(result)


def _windy_expert_action(env: UnifiedControlEnv, observation) -> np.ndarray:
    """High-success PD expert used only to generate paired offline data.

    The default gains are deliberately wind-independent, so labels are a
    deterministic function of exactly the state and goal visible to the Action
    Adapter. They reach about 98% over a broad seed audit. An optional wind
    feed-forward gain remains available for expert ablations but is zero by
    default and never enters the learned method.
    """

    state = observation.control_state
    position, velocity = state[:2], state[2:4]
    displacement = observation.goal - position
    paired = dict(getattr(env, "config", {}).get("paired_data", {}))
    kp = float(paired.get("expert_kp", 2.0))
    kd = float(paired.get("expert_kd", 0.8))
    kw = float(paired.get("expert_kw", 1.0))
    wind = np.zeros_like(displacement)
    if kw:
        current_wind = getattr(env, "current_wind", None)
        if callable(current_wind):
            wind = np.asarray(current_wind(), np.float32)
    return np.clip(
        kp * displacement - kd * velocity - kw * wind,
        env.action_low,
        env.action_high,
    ).astype(np.float32)


def _windy_goal_directed_action(
    env: UnifiedControlEnv,
    observation,
    rng: np.random.Generator,
    mean_scale: float,
    std_scale: float,
) -> np.ndarray:
    direction = np.asarray(observation.goal[:2] - observation.control_state[:2], np.float32)
    norm = float(np.linalg.norm(direction))
    unit = np.zeros(2, np.float32) if norm < 1e-6 else direction / norm
    maximum = float(np.max(np.abs(env.action_high)))
    action = rng.normal(
        loc=unit * maximum * float(mean_scale),
        scale=max(maximum * float(std_scale), 1e-6),
        size=2,
    )
    return np.clip(action, env.action_low, env.action_high).astype(np.float32)


def _windy_action_only_episode(
    env: UnifiedControlEnv, seed: int, horizon: int
) -> dict[str, np.ndarray]:
    """Legacy FA-Planner action data: one constant goal-directed command."""

    observation = env.reset(seed)
    # The action-only prior captures control regularity, not wind response.
    env.region_w0[...] = 0.0
    env.region_k[...] = 0.0
    direction = observation.goal[:2] - observation.control_state[:2]
    norm = float(np.linalg.norm(direction))
    unit = np.array([1.0, 0.0], np.float32) if norm < 1e-6 else direction / norm
    command = np.clip(
        float(np.max(np.abs(env.action_high))) * unit,
        env.action_low,
        env.action_high,
    ).astype(np.float32)
    actions = []
    for _ in range(int(horizon)):
        actions.append(command.copy())
        transition = env.step(command)
        if bool(transition.info.get("success", False)) or transition.truncated:
            break
    action_array = np.asarray(actions, np.float32)
    return {
        "actions": action_array,
        "sequence_length": np.asarray(len(action_array), np.int64),
        "action_low": env.action_low.astype(np.float32),
        "action_high": env.action_high.astype(np.float32),
    }


def _fetch_expert_action(
    env: UnifiedControlEnv, task: str, observation
) -> np.ndarray:
    """Collision-aware Cartesian expert for Push/Slide paired data.

    It first lifts the gripper, moves behind the object relative to the goal,
    descends, and then pushes or strikes.  The controller is deliberately
    outside the learned method and is never available during evaluation.
    """

    state = observation.control_state
    gripper = state[:3]
    obj = state[25:28]
    direction = observation.goal[:2] - obj[:2]
    paired_config = dict(getattr(env, "config", {}).get("paired_data", {}))
    if task == "fetch_slide":
        # FetchSlide's contact geometry systematically attenuates motion along
        # the world y axis.  Calibrating the data-collection controller keeps
        # the puck trajectory aligned with the requested goal; this controller
        # is never available to the learned policy at evaluation time.
        direction[1] *= float(paired_config.get("aim_y_gain", 1.0))
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    default_offset = 0.075 if task == "fetch_slide" else 0.055
    offset = float(paired_config.get("approach_offset", default_offset))
    behind = obj[:2] - offset * direction
    safe_height = obj[2] + 0.10
    relative_xy = gripper[:2] - obj[:2]
    longitudinal = float(np.dot(relative_xy, direction))
    lateral = float(
        abs(relative_xy[0] * direction[1] - relative_xy[1] * direction[0])
    )
    # Once the gripper has descended on the approach line, commit through the
    # object.  Checking the distance to ``behind`` first made the old
    # controller reverse immediately after beginning a strike, producing an
    # oscillatory and needlessly multimodal supervision signal.
    if (
        gripper[2] <= obj[2] + 0.035
        and longitudinal >= -offset - 0.02
        and longitudinal <= 0.025
        and lateral < 0.045
    ):
        if task == "fetch_slide":
            xyz = np.r_[
                direction,
                np.clip(10.0 * (obj[2] - gripper[2]), -1.0, 1.0),
            ]
            return np.r_[xyz, 0.0].astype(np.float32)
        target = np.r_[obj[:2] + 0.07 * direction, obj[2] + 0.01]
        return np.r_[
            np.clip(12.0 * (target - gripper), -1.0, 1.0), 0.0
        ].astype(np.float32)
    if gripper[2] < safe_height - 0.025 and np.linalg.norm(gripper[:2] - behind) > 0.04:
        target = np.r_[gripper[:2], safe_height]
    elif np.linalg.norm(gripper[:2] - behind) > 0.025:
        target = np.r_[behind, safe_height]
    elif gripper[2] > obj[2] + 0.018:
        target = np.r_[behind, obj[2] + 0.008]
    elif task == "fetch_slide":
        xyz = np.r_[direction, np.clip(10.0 * (obj[2] - gripper[2]), -1.0, 1.0)]
        return np.r_[xyz, 0.0].astype(np.float32)
    else:
        target = np.r_[obj[:2] + 0.07 * direction, obj[2] + 0.01]
    return np.r_[np.clip(12.0 * (target - gripper), -1.0, 1.0), 0.0].astype(np.float32)


def paired_expert_action(env: UnifiedControlEnv, observation) -> np.ndarray:
    """Return an environment-specific data-collection action.

    Unsupported tasks fall back to zero residual action.  For Humanoid Stand
    and Balance this means the shared nominal controller remains active.
    """

    task = str(getattr(env, "task", ""))
    if task == "" and env.__class__.__name__ == "WindyControlEnv":
        return _windy_expert_action(env, observation)
    if task in {"fetch_slide", "fetch_push"}:
        return _fetch_expert_action(env, task, observation)
    custom = getattr(env, "paired_expert_action", None)
    if callable(custom):
        return np.asarray(custom(observation), np.float32)
    return np.zeros_like(env.action_low, dtype=np.float32)


def generate_episode(
    env: UnifiedControlEnv,
    kind: DatasetKind | str,
    seed: int,
    horizon: int,
    paired_config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    kind = DatasetKind(kind)
    rng = np.random.default_rng(seed)
    if kind is DatasetKind.ACTION_ONLY:
        if env.__class__.__name__ == "WindyControlEnv":
            return _windy_action_only_episode(env, seed, horizon)
        actions = ou_actions(rng, horizon, env.action_low, env.action_high)
        # No environment-derived field is returned or retained.
        return {
            "actions": actions,
            "sequence_length": np.asarray(horizon, np.int64),
            "action_low": env.action_low.astype(np.float32),
            "action_high": env.action_high.astype(np.float32),
        }
    if kind is DatasetKind.STATE_ONLY:
        prepare = getattr(env, "prepare_passive_episode", None)
        if callable(prepare):
            prepare(rng)
    observation = (
        env.reset(seed, start_mode="random")
        if kind is DatasetKind.STATE_ONLY
        and env.__class__.__name__ == "WindyControlEnv"
        else env.reset(seed)
    )
    rgb: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions_out: list[np.ndarray] = []
    next_rgb: list[np.ndarray] = []
    next_proprio: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    goals: list[np.ndarray] = []
    action_is_expert: list[bool] = []
    candidate_actions = ou_actions(rng, horizon, env.action_low, env.action_high)
    paired_config = dict(paired_config or {})
    expert_fraction = float(paired_config.get("expert_fraction", 0.0))
    if not 0.0 <= expert_fraction <= 1.0:
        raise ValueError("paired_data.expert_fraction must be in [0, 1]")
    use_expert = kind is DatasetKind.PAIRED and rng.random() < expert_fraction
    is_windy = env.__class__.__name__ == "WindyControlEnv"
    windy_paired = kind is DatasetKind.PAIRED and is_windy
    windy_passive = kind is DatasetKind.STATE_ONLY and is_windy
    previous_termination_mode = getattr(env, "terminate_on_success", True)
    if windy_paired or windy_passive:
        env.terminate_on_success = False
    expert_noise = float(paired_config.get("expert_noise", 0.0))
    expert_noise_process = ou_actions(
        rng,
        horizon,
        np.full_like(env.action_low, -1.0),
        np.full_like(env.action_high, 1.0),
        sigma=expert_noise,
    )
    for step in range(horizon):
        rgb.append(observation.rgb)
        proprio.append(observation.proprio)
        states.append(observation.control_state)
        masks.append(observation.state_mask)
        if kind is DatasetKind.STATE_ONLY:
            action = np.zeros_like(env.action_low)
        elif use_expert:
            action = np.clip(
                paired_expert_action(env, observation) + expert_noise_process[step],
                env.action_low,
                env.action_high,
            ).astype(np.float32)
        elif windy_paired:
            action = _windy_goal_directed_action(
                env,
                observation,
                rng,
                float(paired_config.get("gaussian_mean_scale", 1.0)),
                float(paired_config.get("gaussian_std_scale", 0.35)),
            )
        else:
            action = candidate_actions[step]
        transition = env.step(action)
        if kind is DatasetKind.PAIRED:
            actions_out.append(action)
            action_is_expert.append(use_expert)
            next_rgb.append(transition.observation.rgb)
            next_proprio.append(transition.observation.proprio)
            next_states.append(transition.observation.control_state)
            next_masks.append(transition.observation.state_mask)
            rewards.append(transition.reward)
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
            goals.append(observation.goal)
        observation = transition.observation
        if transition.done:
            break
    if windy_paired or windy_passive:
        env.terminate_on_success = previous_termination_mode
    length = len(states)
    if use_expert and bool(paired_config.get("successful_expert_only", False)):
        threshold = float(paired_config.get("success_reward_threshold", 0.0))
        first_success = next(
            (index for index, reward in enumerate(rewards) if reward >= threshold),
            None,
        )
        if first_success is None:
            # Keep failed trajectories for State Adapter dynamics coverage, but
            # do not present their actions as expert targets.
            action_is_expert = [False] * len(action_is_expert)
        elif bool(paired_config.get("expert_until_first_success", False)):
            # Goal-conditioned Fetch tasks do not terminate on success.  Later
            # actions chase an already successful, overshooting object and are
            # not part of the behavior that the evaluation policy must learn.
            action_is_expert = [
                is_expert and index <= first_success
                for index, is_expert in enumerate(action_is_expert)
            ]
    common = {
        "rgb": _stack(rgb, np.uint8),
        "proprio": _stack(proprio, np.float32),
        "control_state": _stack(states, np.float32),
        "state_mask": _stack(masks, bool),
        "sequence_length": np.asarray(length, np.int64),
    }
    if kind is DatasetKind.STATE_ONLY:
        return common
    return {
        **common,
        "actions": _stack(actions_out, np.float32),
        "next_rgb": _stack(next_rgb, np.uint8),
        "next_proprio": _stack(next_proprio, np.float32),
        "next_control_state": _stack(next_states, np.float32),
        "next_state_mask": _stack(next_masks, bool),
        "rewards": _stack(rewards, np.float32),
        "terminated": _stack(terminated, bool),
        "truncated": _stack(truncated, bool),
        "goals": _stack(goals, np.float32),
        "action_is_expert": _stack(action_is_expert, bool),
    }


def generate_relabelled_episode(
    env: UnifiedControlEnv,
    policy: Callable,
    seed: int,
    horizon: int,
    *,
    stop_on_success: bool = True,
) -> dict[str, np.ndarray]:
    """Collect on-policy transitions and relabel their actions with the expert.

    ``actions`` always records what affected the simulator, preserving the
    State Adapter and Action Prior history contracts.  ``expert_actions`` is a
    separate maximum-likelihood target for the Action Adapter.  Keeping those
    fields distinct prevents the common DAgger bookkeeping bug where the world
    model is trained on an action that was never executed.
    """

    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    observation = env.reset(int(seed))
    rgb: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    next_rgb: list[np.ndarray] = []
    next_proprio: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    goals: list[np.ndarray] = []
    for _ in range(int(horizon)):
        rgb.append(observation.rgb)
        proprio.append(observation.proprio)
        states.append(observation.control_state)
        masks.append(observation.state_mask)
        goals.append(observation.goal)
        expert_actions.append(paired_expert_action(env, observation))
        policy_output = policy(observation)
        action = policy_output[0] if isinstance(policy_output, tuple) else policy_output
        action = np.clip(
            np.asarray(action, np.float32), env.action_low, env.action_high
        )
        transition = env.step(action)
        actions.append(action)
        next_rgb.append(transition.observation.rgb)
        next_proprio.append(transition.observation.proprio)
        next_states.append(transition.observation.control_state)
        next_masks.append(transition.observation.state_mask)
        rewards.append(transition.reward)
        terminated.append(transition.terminated)
        truncated.append(transition.truncated)
        observation = transition.observation
        if transition.done or (
            stop_on_success and bool(transition.info.get("success", False))
        ):
            break
    length = len(actions)
    return {
        "rgb": _stack(rgb, np.uint8),
        "proprio": _stack(proprio, np.float32),
        "control_state": _stack(states, np.float32),
        "state_mask": _stack(masks, bool),
        "actions": _stack(actions, np.float32),
        "expert_actions": _stack(expert_actions, np.float32),
        "next_rgb": _stack(next_rgb, np.uint8),
        "next_proprio": _stack(next_proprio, np.float32),
        "next_control_state": _stack(next_states, np.float32),
        "next_state_mask": _stack(next_masks, bool),
        "rewards": _stack(rewards, np.float32),
        "terminated": _stack(terminated, bool),
        "truncated": _stack(truncated, bool),
        "goals": _stack(goals, np.float32),
        "action_is_expert": np.ones(length, bool),
        "sequence_length": np.asarray(length, np.int64),
    }


def generate_dataset(
    env: UnifiedControlEnv,
    output: str,
    kind: DatasetKind | str,
    episodes: int,
    horizon: int,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
    paired_config: Mapping[str, Any] | None = None,
) -> None:
    writer = EpisodeWriter(output, kind, metadata)
    for episode in range(int(episodes)):
        writer.write(
            episode,
            generate_episode(
                env,
                kind,
                seed + episode,
                horizon,
                paired_config=paired_config,
            ),
        )
