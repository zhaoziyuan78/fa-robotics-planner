"""Offline goal-conditioned IQL baseline over paired trajectories."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fa_robotics_planner.baselines.offline_data import (
    achieved_goal_slice,
    load_paired_data,
    stack_transitions,
)

from fa_robotics_planner.envs.unified import UnifiedControlEnv


def _mlp(input_dim: int, output_dim: int, hidden: int, depth: int):
    import torch.nn as nn

    layers: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(int(depth)):
        layers.extend((nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.SiLU()))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _expectile_loss(difference, expectile: float):
    import torch

    weight = torch.where(difference > 0, expectile, 1.0 - expectile)
    return (weight * difference.square()).mean()


class OfflineGCRL:
    """Container around actor, twin critics, and expectile value network."""

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden: int, depth: int):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                observation_dim = state_dim + goal_dim
                self.actor_backbone = _mlp(observation_dim, hidden, hidden, depth)
                self.actor_mean = nn.Linear(hidden, action_dim)
                self.actor_log_std = nn.Linear(hidden, action_dim)
                self.q1 = _mlp(observation_dim + action_dim, 1, hidden, depth)
                self.q2 = _mlp(observation_dim + action_dim, 1, hidden, depth)
                self.value = _mlp(observation_dim, 1, hidden, depth)

            def actor_parameters(self, observation):
                hidden_value = self.actor_backbone(observation)
                return self.actor_mean(hidden_value), self.actor_log_std(hidden_value).clamp(-5, 2)

            def log_prob(self, observation, action):
                import torch

                mean, log_std = self.actor_parameters(observation)
                clipped = action.clamp(-0.999999, 0.999999)
                before_tanh = torch.atanh(clipped)
                normal = -0.5 * (
                    ((before_tanh - mean) / log_std.exp()).square()
                    + 2.0 * log_std
                    + np.log(2.0 * np.pi)
                )
                correction = torch.log(1.0 - clipped.square() + 1e-6)
                return (normal - correction).sum(dim=-1)

            def act(self, observation):
                import torch

                mean, _ = self.actor_parameters(observation)
                return torch.tanh(mean)

        self.model = Model()


def _normalization(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-4).astype(np.float32)


def train_gcrl(
    data_path: str | Path,
    environment_config: Mapping[str, Any],
    config: Mapping[str, Any],
    output: str | Path,
    offline_transitions: int,
    seed: int,
    checkpoint_directory: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Train goal-conditioned IQL using only a static paired replay dataset."""

    import torch
    import torch.nn.functional as functional
    from tqdm.auto import trange

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device(str(config.get("device", "cuda")) if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data = load_paired_data(data_path, int(offline_transitions))
    arrays = stack_transitions(data)
    states = arrays["states"]
    next_states = arrays["next_states"]
    goals = arrays["goals"]
    actions = np.clip(arrays["actions"], -1.0, 1.0)
    dones = arrays["dones"]
    rewards = arrays["rewards"].copy()
    goal_slice = achieved_goal_slice(dict(environment_config), data.goal_dim)
    goal_reward_type = str(config.get("goal_reward_type", "dense"))
    if goal_reward_type not in {"dense", "sparse", "native"}:
        raise ValueError("baseline.goal_reward_type must be dense, sparse, or native")
    if goal_slice is not None and goal_reward_type != "native":
        distances = np.linalg.norm(next_states[:, goal_slice] - goals, axis=-1)
        radius = float(environment_config.get("success_radius", 0.05))
        rewards = -distances if goal_reward_type == "dense" else np.where(distances <= radius, 0.0, -1.0)
    reward_scale = float(config.get("reward_scale", 0.0))
    if reward_scale <= 0:
        reward_scale = 1.0 / max(float(np.std(rewards)), 1.0)
    rewards = rewards.astype(np.float32) * reward_scale

    state_mean, state_std = _normalization(np.concatenate((states, next_states), axis=0))
    if data.goal_dim:
        goal_mean, goal_std = _normalization(goals)
    else:
        goal_mean = np.empty(0, np.float32)
        goal_std = np.empty(0, np.float32)

    def observations(state_values: np.ndarray, goal_values: np.ndarray) -> np.ndarray:
        normalized_state = (state_values - state_mean) / state_std
        if not data.goal_dim:
            return normalized_state.astype(np.float32)
        normalized_goal = (goal_values - goal_mean) / goal_std
        return np.concatenate((normalized_state, normalized_goal), axis=-1).astype(np.float32)

    model_config = {
        "state_dim": data.state_dim,
        "goal_dim": data.goal_dim,
        "action_dim": data.action_dim,
        "hidden": int(config.get("hidden", 256)),
        "depth": int(config.get("depth", 3)),
    }
    container = OfflineGCRL(**model_config)
    model = container.model.to(device)
    target_q1 = copy.deepcopy(model.q1).to(device).eval()
    target_q2 = copy.deepcopy(model.q2).to(device).eval()
    for parameter in [*target_q1.parameters(), *target_q2.parameters()]:
        parameter.requires_grad_(False)
    lr = float(config.get("learning_rate", 3e-4))
    q_optimizer = torch.optim.AdamW([*model.q1.parameters(), *model.q2.parameters()], lr=lr)
    value_optimizer = torch.optim.AdamW(model.value.parameters(), lr=lr)
    actor_optimizer = torch.optim.AdamW(
        [*model.actor_backbone.parameters(), *model.actor_mean.parameters(), *model.actor_log_std.parameters()],
        lr=lr,
    )
    batch_size = min(int(config.get("batch_size", 256)), data.transition_count)
    epochs = int(config.get("epochs", 50))
    configured_updates = config.get("gradient_updates")
    updates = int(configured_updates) if configured_updates is not None else epochs * max(1, int(np.ceil(data.transition_count / batch_size)))
    rng = np.random.default_rng(int(seed))
    discount = float(config.get("discount", 0.99))
    expectile = float(config.get("expectile", 0.7))
    temperature = float(config.get("advantage_temperature", 3.0))
    maximum_weight = float(config.get("max_advantage_weight", 100.0))
    tau = float(config.get("target_tau", 0.005))
    grad_clip = float(config.get("grad_clip", 10.0))
    history: dict[str, list[float]] = {name: [] for name in ("q", "value", "actor", "advantage")}
    episode_ends = np.empty(data.transition_count, np.int64)
    cursor = 0
    for episode in data.episodes:
        episode_ends[cursor : cursor + episode.length] = cursor + episode.length
        cursor += episode.length
    her_ratio = float(config.get("her_ratio", 0.5 if data.goal_dim else 0.0))
    progress = trange(updates, desc="offline GC-IQL", leave=False)
    model.train()
    for _ in progress:
        indices = rng.integers(0, data.transition_count, size=batch_size)
        batch_goals = goals[indices].copy()
        batch_rewards = rewards[indices].copy()
        if goal_slice is not None and her_ratio > 0:
            relabel = rng.random(batch_size) < her_ratio
            offsets = (rng.random(batch_size) * (episode_ends[indices] - indices)).astype(np.int64)
            future = indices + offsets
            batch_goals[relabel] = next_states[future[relabel], goal_slice]
            distance = np.linalg.norm(next_states[indices, goal_slice] - batch_goals, axis=-1)
            radius = float(environment_config.get("success_radius", 0.05))
            relabelled = -distance if goal_reward_type == "dense" else np.where(distance <= radius, 0.0, -1.0)
            batch_rewards[relabel] = relabelled[relabel] * reward_scale
        obs = torch.as_tensor(observations(states[indices], batch_goals), device=device)
        next_obs = torch.as_tensor(observations(next_states[indices], batch_goals), device=device)
        action = torch.as_tensor(actions[indices], device=device)
        reward = torch.as_tensor(batch_rewards, device=device)
        done = torch.as_tensor(dones[indices], device=device)

        with torch.no_grad():
            q_target = reward + discount * (1.0 - done) * model.value(next_obs).squeeze(-1)
        q1 = model.q1(torch.cat((obs, action), dim=-1)).squeeze(-1)
        q2 = model.q2(torch.cat((obs, action), dim=-1)).squeeze(-1)
        q_loss = functional.mse_loss(q1, q_target) + functional.mse_loss(q2, q_target)
        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.q1.parameters(), *model.q2.parameters()], grad_clip)
        q_optimizer.step()

        with torch.no_grad():
            target_q = torch.minimum(
                target_q1(torch.cat((obs, action), dim=-1)).squeeze(-1),
                target_q2(torch.cat((obs, action), dim=-1)).squeeze(-1),
            )
        value = model.value(obs).squeeze(-1)
        value_loss = _expectile_loss(target_q - value, expectile)
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value.parameters(), grad_clip)
        value_optimizer.step()

        with torch.no_grad():
            advantage = target_q - model.value(obs).squeeze(-1)
            weight = torch.exp(temperature * advantage).clamp(max=maximum_weight)
        actor_loss = -(weight * model.log_prob(obs, action)).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.actor_backbone.parameters(), *model.actor_mean.parameters(), *model.actor_log_std.parameters()],
            grad_clip,
        )
        actor_optimizer.step()
        with torch.no_grad():
            for source, target in ((model.q1, target_q1), (model.q2, target_q2)):
                for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                    target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
        values = (q_loss, value_loss, actor_loss, advantage.mean())
        for name, value_item in zip(history, values):
            history[name].append(float(value_item.detach()))
        if len(history["q"]) % max(1, updates // 20) == 0:
            progress.set_postfix(q=f"{history['q'][-1]:.3g}", actor=f"{history['actor'][-1]:.3g}")

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_target = Path(checkpoint_directory) if checkpoint_directory else target / "checkpoint"
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_target / "final_model.pt"
    statistics = {
        "state_mean": state_mean,
        "state_std": state_std,
        "goal_mean": goal_mean,
        "goal_std": goal_std,
    }
    torch.save(
        {
            "algorithm": "offline_goal_conditioned_iql",
            "model_config": model_config,
            "model": model.state_dict(),
            "statistics": statistics,
            "reward_scale": reward_scale,
        },
        checkpoint,
    )
    diagnostics = {
        "gradient_updates": updates,
        "training_epochs": epochs if configured_updates is None else None,
        "loss_first": {name: float(np.mean(values[: min(20, len(values))])) for name, values in history.items()},
        "loss_last": {name: float(np.mean(values[-min(20, len(values)) :])) for name, values in history.items()},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "offline_transitions": data.transition_count,
        "training_environment_steps": 0,
        "dataset": str(data.root),
    }
    (target / "training_history.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return checkpoint, diagnostics


def evaluate_gcrl(
    checkpoint: str | Path,
    env: UnifiedControlEnv,
    episode_horizon: int,
    seeds: list[int],
    video_path: str | Path | None = None,
) -> list[dict[str, object]]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device)
    container = OfflineGCRL(**payload["model_config"])
    model = container.model.to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    statistics = payload["statistics"]
    state_mean = np.asarray(statistics["state_mean"], np.float32)
    state_std = np.asarray(statistics["state_std"], np.float32)
    goal_mean = np.asarray(statistics["goal_mean"], np.float32)
    goal_std = np.asarray(statistics["goal_std"], np.float32)
    results = []
    for episode_index, seed in enumerate(seeds):
        observation = env.reset(int(seed))
        frames = [env.render().copy()] if episode_index == 0 and video_path else []
        frame_metrics = [{"reward": 0.0, "success": False}] if frames else []
        episode_return = 0.0
        step_rewards: list[float] = []
        success = False
        for step in range(int(episode_horizon)):
            state = (observation.control_state - state_mean) / state_std
            if observation.goal.size:
                goal = (observation.goal - goal_mean) / goal_std
                vector = np.concatenate((state, goal))
            else:
                vector = state
            with torch.no_grad():
                normalized_action = model.act(
                    torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
                )[0].cpu().numpy()
            action = env.action_low + 0.5 * (normalized_action + 1.0) * (env.action_high - env.action_low)
            transition = env.step(action)
            observation = transition.observation
            native_reward = float(transition.reward)
            episode_return += native_reward
            step_rewards.append(native_reward)
            success = success or bool(transition.info.get("success", False))
            if frames:
                frames.append(env.render().copy())
                frame_metrics.append(
                    {"reward": native_reward, "success": bool(success)}
                )
            if transition.done:
                break
        achieved = np.asarray(env.get_achieved_goal(), np.float32)
        desired = np.asarray(env.get_goal(), np.float32)
        results.append(
            {
                "seed": int(seed),
                "return": episode_return,
                "rewards": step_rewards,
                "success": success,
                "episode_length": step + 1,
                "final_goal_distance": float(np.linalg.norm(achieved - desired))
                if achieved.size and achieved.shape == desired.shape
                else float("nan"),
            }
        )
        if frames:
            from fa_robotics_planner.visualization import save_rollout_video

            save_rollout_video(
                frames,
                frame_metrics,
                video_path,
                "Offline GC-IQL",
                type(env).__name__,
                int(seed),
                int(episode_horizon),
            )
    return results
