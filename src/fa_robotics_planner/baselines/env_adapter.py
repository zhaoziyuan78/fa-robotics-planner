"""Gymnasium adapters shared by independent baseline implementations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fa_robotics_planner.envs.unified import ObservationBundle, UnifiedControlEnv


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
                info.update(
                    success=success,
                    is_success=success,
                    success_subtasks=float(info.get("success_subtasks", success)),
                    is_terminal=bool(result.terminated),
                )
                achieved = np.asarray(self.unified.get_achieved_goal(), np.float32)
                desired = np.asarray(self.unified.get_goal(), np.float32)
                if achieved.size and achieved.shape == desired.shape:
                    info["goal_distance"] = float(np.linalg.norm(achieved - desired))
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
                    float(result.reward),
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
