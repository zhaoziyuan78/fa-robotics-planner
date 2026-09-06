"""Gymnasium adapters shared by independent baseline implementations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fa_robotics_planner.envs.unified import ObservationBundle, UnifiedControlEnv


def resolve_task_setting(
    baseline_config: Mapping[str, object],
    environment: str,
    name: str,
    default: object,
) -> object:
    """Resolve ``name_by_task[environment]`` before the shared default."""

    task_overrides = baseline_config.get(f"{name}_by_task", {})
    if isinstance(task_overrides, Mapping) and environment in task_overrides:
        return task_overrides[environment]
    return baseline_config.get(name, default)


def flatten_observation(observation: ObservationBundle) -> np.ndarray:
    """Return a non-redundant public state view for vector baselines.

    ``control_state`` already contains proprioception (and achieved goal where
    applicable). Concatenating proprioception again plus a task-constant padding
    mask more than doubled Fetch's input dimension and made the low-budget
    baselines needlessly hard to optimize.
    """

    return np.concatenate(
        (
            observation.control_state,
            observation.goal,
        )
    ).astype(np.float32, copy=False)


def resolve_training_reward_mode(
    baseline_config: Mapping[str, object], environment: str
) -> str:
    """Resolve the reward exposed while an online baseline is training.

    Sparse success-only rewards are a particularly poor diagnostic for short
    online-control runs: a learner that has never reached the goal receives an
    all-zero replay buffer.  Baselines can therefore opt individual tasks into
    a goal-distance reward computed solely from the public achieved and desired
    goals.  Evaluation always constructs an adapter in ``native`` mode.
    """

    mode = str(
        resolve_task_setting(
            baseline_config, environment, "training_reward", "native"
        )
    )
    if mode not in {"native", "dense_goal", "goal_progress"}:
        raise ValueError(
            "Unknown baseline training reward "
            f"{mode!r}; expected native, dense_goal, or goal_progress"
        )
    return mode


class FlatObservationEnvAdapter:
    """Expose a :class:`UnifiedControlEnv` as a normalized Gymnasium env.

    All algorithms see the same public observation bundle and residual action
    interface. Native actions are reached through one fixed affine map, which
    also handles environments whose bounds are not already ``[-1, 1]``.
    """

    def __new__(
        cls,
        env: UnifiedControlEnv,
        episode_horizon: int,
        initial_seed: int = 0,
        record_video_path: str | None = None,
        record_method: str = "baseline",
        record_task: str = "task",
        reward_mode: str = "native",
        goal_progress_scale: float = 1.0,
    ):
        import gymnasium as gym

        class Adapter(gym.Env):
            metadata = {"render_modes": ["rgb_array"]}

            def __init__(self) -> None:
                self.unified = env
                self.max_episode_steps = int(episode_horizon)
                self._next_seed = int(initial_seed)
                self._step = 0
                self._record_video_path = record_video_path
                self._record_method = str(record_method)
                self._record_task = str(record_task)
                self.reward_mode = str(reward_mode)
                if self.reward_mode not in {
                    "native",
                    "dense_goal",
                    "goal_progress",
                }:
                    raise ValueError(
                        "reward_mode must be native, dense_goal, or goal_progress"
                    )
                self.goal_progress_scale = float(goal_progress_scale)
                if self.goal_progress_scale <= 0:
                    raise ValueError("goal_progress_scale must be positive")
                self._previous_goal_distance: float | None = None
                self._recorded_video = False
                self._video_frames: list[np.ndarray] = []
                self._video_metrics: list[dict[str, object]] = []
                self._video_seed = int(initial_seed)
                initial = self.unified.reset(self._next_seed)
                state = flatten_observation(initial)
                self.observation_space = gym.spaces.Box(
                    -np.inf, np.inf, state.shape, dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    -1.0, 1.0, env.action_low.shape, dtype=np.float32
                )
                self._action_low = np.asarray(env.action_low, np.float32)
                self._action_high = np.asarray(env.action_high, np.float32)

            def _native_action(self, action: np.ndarray) -> np.ndarray:
                normalized = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
                return self._action_low + 0.5 * (normalized + 1.0) * (
                    self._action_high - self._action_low
                )

            def reset(self, *, seed=None, options=None):
                del options
                if seed is None:
                    seed = self._next_seed
                    self._next_seed += 1
                else:
                    seed = int(seed)
                    self._next_seed = seed + 1
                self._step = 0
                observation = self.unified.reset(int(seed))
                achieved = np.asarray(self.unified.get_achieved_goal(), np.float32)
                desired = np.asarray(self.unified.get_goal(), np.float32)
                self._previous_goal_distance = (
                    float(np.linalg.norm(achieved - desired))
                    if achieved.size and achieved.shape == desired.shape
                    else None
                )
                if self._record_video_path and not self._recorded_video:
                    self._video_seed = int(seed)
                    self._video_frames = [self.unified.render().copy()]
                    self._video_metrics = [{"reward": 0.0, "success": False}]
                return flatten_observation(observation), {"seed": int(seed)}

            def step(self, action):
                result = self.unified.step(self._native_action(action))
                self._step += 1
                truncated = bool(
                    result.truncated or self._step >= self.max_episode_steps
                )
                info = dict(result.info)
                success = float(info.get("success", info.get("is_success", False)))
                native_reward = float(result.reward)
                info.update(
                    success=success,
                    is_success=success,
                    success_subtasks=float(info.get("success_subtasks", success)),
                    is_terminal=bool(result.terminated),
                )
                achieved = np.asarray(self.unified.get_achieved_goal(), np.float32)
                desired = np.asarray(self.unified.get_goal(), np.float32)
                reward = native_reward
                if achieved.size and achieved.shape == desired.shape:
                    goal_distance = float(np.linalg.norm(achieved - desired))
                    info["goal_distance"] = goal_distance
                    if self.reward_mode == "dense_goal":
                        reward = -goal_distance
                    elif self.reward_mode == "goal_progress":
                        if self._previous_goal_distance is None:
                            raise RuntimeError(
                                "goal_progress reward has no reset distance"
                            )
                        reward = self.goal_progress_scale * (
                            self._previous_goal_distance - goal_distance
                        ) + native_reward
                    self._previous_goal_distance = goal_distance
                elif self.reward_mode in {"dense_goal", "goal_progress"}:
                    raise ValueError(
                        "dense_goal reward requires matching non-empty achieved and desired goals"
                    )
                info["native_reward"] = native_reward
                info["training_reward"] = float(reward)
                if self._record_video_path and not self._recorded_video:
                    self._video_frames.append(self.unified.render().copy())
                    self._video_metrics.append(
                        {"reward": float(result.reward), "success": bool(success)}
                    )
                    if result.terminated or truncated:
                        from fa_robotics_planner.visualization import save_rollout_video

                        save_rollout_video(
                            self._video_frames,
                            self._video_metrics,
                            self._record_video_path,
                            self._record_method,
                            self._record_task,
                            self._video_seed,
                            self.max_episode_steps,
                        )
                        self._recorded_video = True
                return (
                    flatten_observation(result.observation),
                    float(reward),
                    bool(result.terminated),
                    truncated,
                    info,
                )

            def render(self):
                return self.unified.render()

            def close(self):
                self.unified.close()

        return Adapter()


def baseline_observation_metadata() -> Mapping[str, object]:
    return {
        "observation_mode": "control_state+goal",
        "uses_rgb": False,
        "uses_simulator_private_state": False,
    }
