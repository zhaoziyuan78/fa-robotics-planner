"""GC-SAC / GC-SAC+HER baseline over the exact unified environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fa_robotics_planner.envs.unified import UnifiedControlEnv


def _resolved_learning_starts(
    configured: int, episode_horizon: int, use_her: bool
) -> int:
    """Delay HER updates until at least one complete episode is available."""

    configured = int(configured)
    if configured < 0:
        raise ValueError("baseline.learning_starts must be non-negative")
    if not use_her:
        return configured
    return max(configured, int(episode_horizon) + 1)


def _resolved_replay_buffer_size(
    configured: int | None, total_steps: int, episode_horizon: int, use_her: bool
) -> int:
    """Keep every transition from this short-budget run addressable by HER.

    SB3 invalidates an entire stored episode as soon as its first transition is
    overwritten.  With a replay buffer close to the episode horizon this can
    temporarily leave no complete episode to sample, producing the misleading
    "before the end of the first episode" exception *after* training has
    already started.  The paper sweeps are small enough that retaining the
    complete run is both cheaper and more robust than allowing wrap-around.
    """

    total_steps = int(total_steps)
    episode_horizon = int(episode_horizon)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if episode_horizon <= 0:
        raise ValueError("episode_horizon must be positive")
    requested = total_steps + 1 if configured is None else int(configured)
    if requested <= 0:
        raise ValueError("baseline.buffer_size must be positive")
    minimum = total_steps + 1 if use_her else 1
    return max(requested, minimum)


class GoalEnvAdapter:
    """Gymnasium GoalEnv-style adapter with vectorized compute_reward."""

    def __new__(
        cls,
        env: UnifiedControlEnv,
        episode_horizon: int,
        use_goal_reward: bool = True,
    ):
        import gymnasium as gym

        class Adapter(gym.Env):
            metadata = {"render_modes": ["rgb_array"]}

            def __init__(self):
                self.unified = env
                self.episode_horizon = int(episode_horizon)
                self.use_goal_reward = bool(use_goal_reward)
                self.step_count = 0
                initial = env.reset(0)
                goal_dim = max(1, initial.goal.size)
                self.goal_dim = goal_dim
                self.observation_space = gym.spaces.Dict(
                    {
                        "observation": gym.spaces.Box(-np.inf, np.inf, initial.control_state.shape, np.float32),
                        "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
                        "desired_goal": gym.spaces.Box(-np.inf, np.inf, (goal_dim,), np.float32),
                    }
                )
                self.action_space = gym.spaces.Box(env.action_low, env.action_high, dtype=np.float32)

            def _goal(self, value: np.ndarray) -> np.ndarray:
                value = np.asarray(value, np.float32).reshape(-1)
                return value if value.size else np.zeros(self.goal_dim, np.float32)

            def _observation(self, bundle):
                return {
                    "observation": bundle.control_state.astype(np.float32),
                    "achieved_goal": self._goal(self.unified.get_achieved_goal()),
                    "desired_goal": self._goal(bundle.goal),
                }

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.step_count = 0
                bundle = self.unified.reset(0 if seed is None else int(seed))
                return self._observation(bundle), {}

            def step(self, action):
                transition = self.unified.step(action)
                self.step_count += 1
                info = dict(transition.info)
                info["is_success"] = float(info.get("success", False))
                truncated = transition.truncated or self.step_count >= self.episode_horizon
                observation = self._observation(transition.observation)
                reward = float(transition.reward)
                if self.use_goal_reward:
                    # Real and HER-relabeled samples must use the same reward
                    # convention.  Windy otherwise mixed native 0/1 rewards
                    # with HER's -1/0 rewards in one replay minibatch.
                    info["native_reward"] = reward
                    reward = float(
                        self.compute_reward(
                            observation["achieved_goal"],
                            observation["desired_goal"],
                            info,
                        )
                    )
                return observation, reward, transition.terminated, truncated, info

            def compute_reward(self, achieved_goal, desired_goal, info):
                distance = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
                radius = float(getattr(self.unified, "success_radius", 0.05))
                return np.where(distance <= radius, 0.0, -1.0).astype(np.float32)

            def render(self):
                return self.unified.render()

            def close(self):
                self.unified.close()

        return Adapter()


def train_gcrl(
    env: UnifiedControlEnv,
    config: Mapping[str, Any],
    output: str | Path,
    total_steps: int,
    seed: int,
    checkpoint_directory: str | Path | None = None,
) -> Path:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_target = (
        Path(checkpoint_directory) if checkpoint_directory else target / "checkpoint"
    )
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    horizon = int(config.get("episode_horizon", 50))
    use_her = bool(config.get("use_her", True))
    gym_env = GoalEnvAdapter(env, horizon, use_goal_reward=use_her)
    configured_learning_starts = int(config.get("learning_starts", 100))
    learning_starts = _resolved_learning_starts(
        configured_learning_starts, horizon, use_her
    )
    configured_buffer_size = config.get("buffer_size")
    buffer_size = _resolved_replay_buffer_size(
        None if configured_buffer_size is None else int(configured_buffer_size),
        int(total_steps),
        horizon,
        use_her,
    )
    if learning_starts != configured_learning_starts:
        print(
            "Adjusted GCRL learning_starts from "
            f"{configured_learning_starts} to {learning_starts}: HER requires "
            "one complete episode before sampling."
        )
    kwargs: dict[str, Any] = {}
    if use_her:
        kwargs.update(
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        )
    model = SAC(
        "MultiInputPolicy",
        gym_env,
        seed=int(seed),
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        batch_size=int(config.get("batch_size", 256)),
        verbose=1,
        **kwargs,
    )
    callback = CheckpointCallback(
        save_freq=max(1, int(config.get("checkpoint_interval", 10000))),
        save_path=str(checkpoint_target),
        name_prefix="gcrl",
    )
    model.learn(total_timesteps=int(total_steps), callback=callback)
    checkpoint = checkpoint_target / "final_model"
    model.save(checkpoint)
    return checkpoint.with_suffix(".zip")


def evaluate_gcrl(
    checkpoint: str | Path,
    env: UnifiedControlEnv,
    episode_horizon: int,
    seeds: list[int],
    use_goal_reward: bool = True,
    video_path: str | Path | None = None,
) -> list[dict[str, float | int | bool]]:
    from stable_baselines3 import SAC

    gym_env = GoalEnvAdapter(
        env, episode_horizon, use_goal_reward=use_goal_reward
    )
    model = SAC.load(checkpoint, env=gym_env)
    results = []
    for episode_index, seed in enumerate(seeds):
        observation, _ = gym_env.reset(seed=int(seed))
        frames = [gym_env.render().copy()] if episode_index == 0 and video_path else []
        frame_metrics = [{"reward": 0.0, "success": False}] if frames else []
        episode_return = 0.0
        goal_sparse_return = 0.0
        success = False
        for step in range(int(episode_horizon)):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = gym_env.step(action)
            goal_sparse_return += float(reward)
            episode_return += float(info.get("native_reward", reward))
            success = success or bool(info.get("is_success", False))
            if frames:
                frames.append(gym_env.render().copy())
                frame_metrics.append(
                    {"reward": float(reward), "success": bool(success)}
                )
            if terminated or truncated:
                break
        achieved = np.asarray(observation["achieved_goal"], np.float32)
        desired = np.asarray(observation["desired_goal"], np.float32)
        results.append(
            {
                "seed": int(seed),
                "return": episode_return,
                "goal_sparse_return": goal_sparse_return,
                "success": success,
                "episode_length": step + 1,
                "final_goal_distance": float(np.linalg.norm(achieved - desired)),
            }
        )
        if frames:
            from fa_robotics_planner.visualization import save_rollout_video

            save_rollout_video(
                frames,
                frame_metrics,
                video_path,
                "GC-SAC+HER" if use_goal_reward else "GC-SAC",
                type(env).__name__,
                int(seed),
                int(episode_horizon),
            )
    return results
