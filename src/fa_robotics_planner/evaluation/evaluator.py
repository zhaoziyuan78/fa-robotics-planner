from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from tqdm import tqdm

from fa_robotics_planner.envs.unified import ObservationBundle, UnifiedControlEnv

from .metrics import summarize_episodes


Policy = Callable[[ObservationBundle], np.ndarray | tuple[np.ndarray, dict[str, Any]]]


def evaluate_policy(
    env: UnifiedControlEnv,
    policy: Policy,
    seeds: Iterable[int],
    episode_horizon: int,
    bootstrap_samples: int = 10000,
    show_progress: bool = True,
    progress_update_interval: int = 10,
    progress_desc: str = "eval",
    video_output: str | Path | None = None,
    video_method: str = "policy",
    video_task: str = "task",
    video_ood_parameters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = [int(seed) for seed in seeds]
    episode_horizon = int(episode_horizon)
    update_interval = max(1, int(progress_update_interval))
    episodes: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    progress = tqdm(
        total=len(seeds) * episode_horizon,
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
        disable=not show_progress,
    )
    try:
        for episode_index, seed in enumerate(seeds):
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                reset_policy()
            observation = env.reset(seed)
            video_frames = (
                [observation.rgb.copy()]
                if episode_index == 0 and video_output is not None
                else []
            )
            video_metrics = (
                [{"reward": 0.0, "success": False}] if video_frames else []
            )
            episode_return = control_energy = planning_time = 0.0
            step_rewards: list[float] = []
            success = False
            time_to_success = None
            forward_calls = sampled = steps_taken = 0
            terminated = truncated = False
            for step in range(episode_horizon):
                started = time.perf_counter()
                policy_output = policy(observation)
                planning_time += time.perf_counter() - started
                if isinstance(policy_output, tuple):
                    action, diagnostics = policy_output
                    forward_calls += int(diagnostics.get("model_forward_calls", 0))
                    sampled += int(diagnostics.get("sampled_action_sequences", 0))
                else:
                    action, diagnostics = policy_output, {}
                action = np.clip(np.asarray(action, np.float32), env.action_low, env.action_high)
                transition = env.step(action)
                terminated, truncated = transition.terminated, transition.truncated
                episode_return += transition.reward
                step_rewards.append(float(transition.reward))
                control_energy += float(np.square(action).sum())
                success = success or bool(transition.info.get("success", False))
                steps_taken = step + 1
                if success and time_to_success is None:
                    time_to_success = steps_taken
                observation = transition.observation
                if video_frames:
                    video_frames.append(observation.rgb.copy())
                    video_metrics.append(
                        {
                            "reward": float(transition.reward),
                            "success": bool(success),
                        }
                    )
                progress.update(1)
                if steps_taken % update_interval == 0 or transition.done:
                    progress.set_postfix(
                        episode=f"{episode_index + 1}/{len(seeds)}",
                        episode_step=f"{steps_taken}/{episode_horizon}",
                        episode_return=f"{episode_return:.2f}",
                        success=int(success),
                        refresh=False,
                    )
                if transition.done:
                    break
            episode_success = getattr(env, "episode_success", None)
            if callable(episode_success):
                success = bool(
                    episode_success(
                        success,
                        episode_return,
                        steps_taken,
                        episode_horizon,
                        terminated,
                        truncated,
                    )
                )
                if success and time_to_success is None:
                    time_to_success = steps_taken
            if steps_taken < episode_horizon:
                progress.update(episode_horizon - steps_taken)
            episodes.append(
                {
                    "seed": seed,
                    "return": float(episode_return),
                    "rewards": step_rewards,
                    "success": bool(success),
                    "time_to_success": time_to_success,
                    "episode_length": steps_taken,
                    "control_energy": control_energy,
                    "planning_latency": planning_time / max(1, steps_taken),
                    "model_forward_calls": forward_calls / max(1, steps_taken),
                    "sampled_action_sequences": sampled / max(1, steps_taken),
                }
            )
            if video_frames:
                from fa_robotics_planner.visualization import save_rollout_video

                save_rollout_video(
                    video_frames,
                    video_metrics,
                    video_output,
                    video_method,
                    video_task,
                    seed,
                    episode_horizon,
                    video_ood_parameters,
                )
    finally:
        progress.close()
    summary = summarize_episodes(episodes, bootstrap_samples)
    wall_seconds = time.perf_counter() - evaluation_started
    evaluated_steps = sum(int(episode["episode_length"]) for episode in episodes)
    planning_seconds = sum(
        float(episode["planning_latency"]) * int(episode["episode_length"])
        for episode in episodes
    )
    summary["evaluation_wall_seconds"] = wall_seconds
    summary["evaluation_steps"] = evaluated_steps
    summary["evaluation_steps_per_second"] = evaluated_steps / max(
        wall_seconds, 1e-12
    )
    summary["planning_steps_per_second"] = evaluated_steps / max(
        planning_seconds, 1e-12
    )
    return episodes, summary
