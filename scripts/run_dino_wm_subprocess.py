"""DINO-WM strategy-A worker over the modern unified environments.

The official visual model and CEM are imported from an external audited source
checkout. Its legacy Gym and ``mujoco-py`` environments are intentionally never
imported into the main planner environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fa_robotics_planner.baselines.env_adapter import resolve_task_setting
from fa_robotics_planner.baselines.offline_data import load_paired_data
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.envs.unified import ObservationBundle
from fa_robotics_planner.evaluation.metrics import (
    summarize_episodes,
    write_episode_metrics,
)
from fa_robotics_planner.utils.seed import seed_everything


OFFICIAL_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"


def _load_official_source(source: str | Path) -> tuple[Path, str]:
    path = Path(source).expanduser().resolve()
    required = [
        path / "models" / "dino.py",
        path / "models" / "visual_world_model.py",
        path / "planning" / "cem.py",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete official DINO-WM checkout: {missing}")
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"DINO-WM source is {commit}; expected audited commit {OFFICIAL_COMMIT}"
        )
    sys.path.insert(0, str(path))
    return path, commit


class DinoPreprocessor:
    def __init__(
        self,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
    ) -> None:
        import torch

        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.proprio_mean = torch.as_tensor(proprio_mean, dtype=torch.float32)
        self.proprio_std = torch.as_tensor(proprio_std, dtype=torch.float32)

    def normalize_actions(self, actions):
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(
            actions.device
        )

    def denormalize_actions(self, actions):
        return actions * self.action_std.to(actions.device) + self.action_mean.to(
            actions.device
        )

    def normalized_action_bounds(self, low: np.ndarray, high: np.ndarray, device):
        import torch

        low_tensor = torch.as_tensor(low, dtype=torch.float32, device=device)
        high_tensor = torch.as_tensor(high, dtype=torch.float32, device=device)
        mean = self.action_mean.to(device)
        std = self.action_std.to(device)
        return (low_tensor - mean) / std, (high_tensor - mean) / std

    def transform_obs(self, observation: dict[str, np.ndarray]):
        import torch
        import torch.nn.functional as functional

        visual = torch.as_tensor(
            np.asarray(observation["visual"]).copy(), dtype=torch.float32
        )
        visual = visual.permute(0, 1, 4, 2, 3) / 255.0
        shape = visual.shape
        visual = functional.interpolate(
            visual.reshape(-1, *shape[2:]),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).reshape(shape[0], shape[1], 3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        visual = (visual - mean) / std
        proprio = torch.as_tensor(
            np.asarray(observation["proprio"]).copy(), dtype=torch.float32
        )
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        return {"visual": visual, "proprio": proprio}


@dataclass
class OfflineTransitions:
    current_rgb: np.ndarray
    current_proprio: np.ndarray
    actions: np.ndarray
    next_rgb: np.ndarray
    next_proprio: np.ndarray
    window_starts: np.ndarray
    history_frames: int


def _load_transitions(
    data_path: str | Path, steps: int, history_frames: int
) -> OfflineTransitions:
    """Read DINO-WM training pairs without constructing or stepping an env."""

    data = load_paired_data(data_path, int(steps), include_rgb=True)
    history_frames = int(history_frames)
    if history_frames < 1:
        raise ValueError("baseline.history_frames must be positive")
    offsets = np.cumsum(
        np.asarray([0, *[episode.length for episode in data.episodes]], np.int64)
    )
    starts = [
        np.arange(offsets[index], offsets[index + 1] - history_frames + 1)
        for index in range(len(data.episodes))
        if data.episodes[index].length >= history_frames
    ]
    if not starts:
        raise ValueError(
            f"No {history_frames}-transition DINO-WM window in {data.root}"
        )
    window_starts = np.concatenate(starts)
    current_rgb = np.concatenate([episode.rgb for episode in data.episodes], axis=0)
    next_rgb = np.concatenate([episode.next_rgb for episode in data.episodes], axis=0)
    actions = np.concatenate([episode.actions for episode in data.episodes], axis=0)
    # A world model predicts task dynamics, not the data-collection policy.
    # Conditioning it on desired_goal lets it infer goal-directed motion even
    # when the candidate action is zero, a severe confound on paired expert
    # data.  The goal is supplied separately as CEM's target observation.
    current_proprio = np.concatenate(
        [episode.states for episode in data.episodes], axis=0
    )
    next_proprio = np.concatenate(
        [episode.next_states for episode in data.episodes], axis=0
    )
    return OfflineTransitions(
        current_rgb,
        current_proprio,
        actions,
        next_rgb,
        next_proprio,
        window_starts.astype(np.int64, copy=False),
        history_frames,
    )


def _build_model(
    proprio_size: int,
    action_size: int,
    device,
    model_settings: dict[str, Any],
):
    from models.dino import DinoV2Encoder
    from models.proprio import ProprioceptiveEmbedding
    from models.visual_world_model import VWorldModel
    from models.vit import ViTPredictor

    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    proprio_encoder = ProprioceptiveEmbedding(
        num_frames=int(model_settings["history_frames"]),
        in_chans=int(proprio_size),
        emb_dim=encoder.emb_dim,
    )
    action_encoder = ProprioceptiveEmbedding(
        num_frames=1, in_chans=int(action_size), emb_dim=encoder.emb_dim
    )
    predictor = ViTPredictor(
        num_patches=198,
        num_frames=int(model_settings["history_frames"]),
        dim=encoder.emb_dim,
        depth=int(model_settings["predictor_depth"]),
        heads=int(model_settings["predictor_heads"]),
        mlp_dim=int(model_settings["predictor_mlp_dim"]),
        dropout=float(model_settings["predictor_dropout"]),
        emb_dropout=0.0,
        pool="mean",
    )
    model = VWorldModel(
        image_size=224,
        num_hist=int(model_settings["history_frames"]),
        num_pred=1,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=encoder.emb_dim,
        action_dim=encoder.emb_dim,
        concat_dim=0,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    )
    if not bool(model_settings.get("train_proprio_encoder", False)):
        # A jointly learned linear proprio target can reduce the loss simply by
        # collapsing its scale.  Planning then pseudo-inverts a nearly singular
        # projection.  A frozen full-rank projection preserves every public
        # structured coordinate while the predictor learns its dynamics.
        for parameter in model.proprio_encoder.parameters():
            parameter.requires_grad = False
    return model.to(device)


def _batch(
    transitions: OfflineTransitions,
    indices: np.ndarray,
    preprocessor: DinoPreprocessor,
    device,
):
    import torch

    history = transitions.history_frames
    offsets = np.arange(history, dtype=np.int64)[None, :]
    transition_indices = indices[:, None] + offsets
    # A H-transition window contains H+1 observations.  This is the official
    # DINO-WM training contract (num_hist=H, num_pred=1); independently sampled
    # one-step pairs silently removed its temporal-context training.
    observation = {
        "visual": np.concatenate(
            (
                transitions.current_rgb[indices, None],
                transitions.next_rgb[transition_indices],
            ),
            axis=1,
        ),
        "proprio": np.concatenate(
            (
                transitions.current_proprio[indices, None],
                transitions.next_proprio[transition_indices],
            ),
            axis=1,
        ),
    }
    observation = {
        name: value.to(device) for name, value in preprocessor.transform_obs(observation).items()
    }
    window_actions = transitions.actions[transition_indices]
    actions = np.concatenate(
        (window_actions, np.zeros_like(window_actions[:, :1])), axis=1
    )
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    return observation, preprocessor.normalize_actions(actions)


def _goal_observation(env_name: str, env, observation: ObservationBundle) -> dict[str, np.ndarray]:
    rgb = observation.rgb.copy()
    proprio = observation.proprio.copy()
    control = observation.control_state.copy()
    goal = observation.goal.copy()
    if env_name == "windy":
        from fa_robotics_planner.envs.rendering import render_windy

        rgb = render_windy(
            observation.rgb.shape[0], goal, np.zeros(2, np.float32), goal, env.region_colors
        )
        proprio = np.zeros_like(proprio)
        control[:2] = goal
        control[2:4] = 0.0
    elif env_name in {"fetch_slide", "fetch_push"} and goal.size:
        if proprio.size >= 6:
            proprio[3 : 3 + goal.size] = goal
        control[: proprio.size] = proprio
        control[proprio.size : proprio.size + goal.size] = goal
        render_goal = getattr(env, "render_goal", None)
        if callable(render_goal):
            rgb = render_goal(goal)
    target = ObservationBundle(rgb, proprio, control, observation.state_mask, goal)
    return {
        "visual": target.rgb[None, None],
        "proprio": target.control_state[None, None],
    }


class _NoopRun:
    def log(self, *args, **kwargs):
        del args, kwargs


def _model_settings(baseline: dict[str, Any]) -> dict[str, Any]:
    """Resolve and persist every architectural choice used by DINO-WM."""

    return {
        "history_frames": int(baseline.get("history_frames", 3)),
        "predictor_depth": int(baseline.get("predictor_depth", 6)),
        "predictor_heads": int(baseline.get("predictor_heads", 16)),
        "predictor_mlp_dim": int(baseline.get("predictor_mlp_dim", 2048)),
        "predictor_dropout": float(baseline.get("predictor_dropout", 0.1)),
        "train_proprio_encoder": bool(
            baseline.get("train_proprio_encoder", False)
        ),
    }


def _decode_proprio_tokens(model, preprocessor, token):
    """Decode the fixed full-rank structured token in native coordinates."""

    import torch

    projection = model.proprio_encoder.patch_embed
    weight = projection.weight[..., 0]
    inverse = torch.linalg.pinv(weight)
    normalized = (token - projection.bias) @ inverse.T
    return normalized * preprocessor.proprio_std.to(token.device) + preprocessor.proprio_mean.to(
        token.device
    )


def _create_planning_objective(model, preprocessor, config, baseline):
    """Build the DINO-WM goal score used by CEM.

    The official latent MSE scores every proprioceptive coordinate equally.
    That is incorrect for Windy: acceleration must first create velocity, but
    comparing that velocity with the zero-velocity goal makes useful actions
    look worse than doing nothing.  The proprio encoder is a one-step linear
    projection, so its predicted token can be decoded exactly with a
    pseudoinverse and scored only on the task's public achieved-goal slice.
    """

    env_name = str(config["env"]["name"])
    objective = str(
        resolve_task_setting(
            baseline, env_name, "planning_objective", "latent"
        )
    )
    if objective == "latent":
        from planning.objectives import create_objective_fn

        return create_objective_fn(
            alpha=float(baseline.get("proprio_alpha", 10.0)), base=1.0
        )
    if objective not in {"structured_goal", "kinematic_goal"}:
        raise ValueError(
            "baseline.planning_objective must be latent, structured_goal, or "
            "kinematic_goal"
        )

    import torch

    achieved_bounds = config["env"].get(
        "action_adapter_achieved_goal_slice",
        config["env"].get("achieved_goal_slice"),
    )
    if achieved_bounds is None:
        raise ValueError(
            "structured_goal DINO-WM planning requires an achieved-goal slice"
        )
    start, stop = (int(value) for value in achieved_bounds)
    dimensions = min(
        stop - start,
        int(config["env"].get("goal_score_dimensions", stop - start)),
    )
    stop = start + dimensions
    visual_weight = float(baseline.get("planning_visual_weight", 0.0))
    velocity_bounds = config["env"].get("velocity_slice")
    if objective == "kinematic_goal" and velocity_bounds is None:
        raise ValueError("kinematic_goal requires env.velocity_slice")
    projection = model.proprio_encoder.patch_embed
    inverse = torch.linalg.pinv(projection.weight[..., 0].detach())
    mean = preprocessor.proprio_mean.to(inverse.device)
    std = preprocessor.proprio_std.to(inverse.device)

    def decode(token):
        normalized = (token - projection.bias) @ inverse.T
        return normalized * std + mean

    def objective_fn(predicted, target):
        predicted_trajectory = decode(predicted["proprio"])
        target_state = decode(target["proprio"][:, -1])
        if objective == "kinematic_goal":
            velocity_start, velocity_stop = map(int, velocity_bounds)
            if velocity_stop - velocity_start < stop - start:
                raise ValueError("env.velocity_slice is smaller than achieved-goal slice")
            dimensions = stop - start
            velocities = predicted_trajectory[
                :, 1:, velocity_start : velocity_start + dimensions
            ]
            predicted_goal = (
                predicted_trajectory[:, 0, start:stop]
                + float(config["env"].get("dt", 1.0)) * velocities.sum(dim=1)
            )
        else:
            predicted_goal = predicted_trajectory[:, -1, start:stop]
        error = predicted_goal - target_state[:, start:stop]
        loss = error.square().mean(dim=-1)
        if visual_weight:
            visual_error = (
                predicted["visual"][:, -1:] - target["visual"]
            ).square()
            loss = loss + visual_weight * visual_error.flatten(1).mean(dim=-1)
        return loss

    return objective_fn


def _evaluate(
    model,
    preprocessor,
    config,
    seed: int,
    baseline: dict[str, Any],
    episodes_count: int,
    video_path: str | Path | None = None,
):
    import torch
    from einops import repeat
    from planning.cem import CEMPlanner
    from utils import move_to_device

    env_name = str(config["env"]["name"])
    if env_name.startswith("humanoid_"):
        raise NotImplementedError(
            "DINO-WM strategy-A smoke is validated for Windy and Fetch; "
            "Humanoid needs a goal-image protocol"
        )
    env = make_env(config)
    task_setting = lambda key, default: resolve_task_setting(
        baseline, env_name, key, default
    )
    training_history_frames = int(model.num_hist)
    planning_history_frames = int(
        task_setting("planning_history_frames", training_history_frames)
    )
    if not 1 <= planning_history_frames <= training_history_frames:
        raise ValueError(
            "baseline.planning_history_frames must be between 1 and "
            "baseline.history_frames"
        )
    # The appended control_state is Markov for every supported environment.
    # Training still uses consecutive histories, while retaining only the most
    # recent latent during autoregressive MPC avoids quadratic attention over
    # redundant predicted histories on the RTX 5000.
    model.num_hist = planning_history_frames

    class BoundedCEMPlanner(CEMPlanner):
        """Official CEM with actions kept inside the environment support.

        The upstream implementation scores unbounded Gaussian samples and only
        the caller clipped the selected action afterwards.  That lets CEM
        exploit world-model predictions for actions the environment will never
        execute.  Bounds are expressed in the same normalized coordinates as
        the offline training actions.
        """

        def __init__(self, *args, action_lower, action_upper, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_lower = action_lower
            self.action_upper = action_upper
            self.encoded_goal = None

        def clear_goal_cache(self):
            self.encoded_goal = None

        def rollout_encoded_observation(self, encoded_observation, actions):
            """Exact ``VWorldModel.rollout`` without re-encoding duplicate RGB."""

            initial_frames = encoded_observation["visual"].shape[1]
            initial_action = actions[:, :initial_frames]
            visual = encoded_observation["visual"]
            proprio = encoded_observation["proprio"].unsqueeze(2)
            action_token = self.wm.encode_act(initial_action).unsqueeze(2)
            latent = torch.cat((visual, proprio, action_token), dim=2)
            future_actions = actions[:, initial_frames:]
            for index in range(future_actions.shape[1]):
                prediction = self.wm.predict(
                    latent[:, -self.wm.num_hist :]
                )
                new_latent = prediction[:, -1:]
                new_latent = self.wm.replace_actions_from_z(
                    new_latent, future_actions[:, index : index + 1]
                )
                latent = torch.cat((latent, new_latent), dim=1)
            prediction = self.wm.predict(latent[:, -self.wm.num_hist :])
            latent = torch.cat((latent, prediction[:, -1:]), dim=1)
            predicted_observation, _ = self.wm.separate_emb(latent)
            return predicted_observation

        def plan(self, obs_0, obs_g, actions=None):
            transformed = move_to_device(
                self.preprocessor.transform_obs(obs_0), self.device
            )
            encoded_observation = self.wm.encode_obs(transformed)
            if self.encoded_goal is None:
                transformed_goal = move_to_device(
                    self.preprocessor.transform_obs(obs_g), self.device
                )
                self.encoded_goal = self.wm.encode_obs(transformed_goal)
            encoded_goal = self.encoded_goal
            mu, sigma = self.init_mu_sigma(obs_0, actions)
            mu, sigma = mu.to(self.device), sigma.to(self.device)
            lower = self.action_lower.to(self.device).view(1, 1, -1)
            upper = self.action_upper.to(self.device).view(1, 1, -1)
            mu = torch.maximum(torch.minimum(mu, upper), lower)
            for _ in range(self.opt_steps):
                for trajectory in range(mu.shape[0]):
                    repeated_observation = {
                        key: repeat(
                            value[trajectory].unsqueeze(0),
                            "1 ... -> n ...",
                            n=self.num_samples,
                        )
                        for key, value in encoded_observation.items()
                    }
                    repeated_goal = {
                        key: repeat(
                            value[trajectory].unsqueeze(0),
                            "1 ... -> n ...",
                            n=self.num_samples,
                        )
                        for key, value in encoded_goal.items()
                    }
                    candidates = (
                        torch.randn(
                            self.num_samples,
                            self.horizon,
                            self.action_dim,
                            device=self.device,
                        )
                        * sigma[trajectory]
                        + mu[trajectory]
                    )
                    candidates = torch.maximum(
                        torch.minimum(candidates, upper), lower
                    )
                    candidates[0] = mu[trajectory]
                    predicted = self.rollout_encoded_observation(
                        repeated_observation, candidates
                    )
                    loss = self.objective_fn(predicted, repeated_goal)
                    elite = candidates[
                        torch.argsort(loss)[: self.topk]
                    ]
                    mu[trajectory] = elite.mean(dim=0)
                    sigma[trajectory] = elite.std(
                        dim=0, unbiased=False
                    ).clamp_min(1e-3)
            return mu, np.full(mu.shape[0], np.inf)

    normalized_low, normalized_high = preprocessor.normalized_action_bounds(
        env.action_low, env.action_high, torch.device("cuda")
    )
    planner = BoundedCEMPlanner(
        horizon=int(task_setting("horizon", 5)),
        topk=int(task_setting("topk", 8)),
        num_samples=int(task_setting("num_samples", 64)),
        var_scale=float(task_setting("var_scale", 1.0)),
        opt_steps=int(task_setting("opt_steps", 5)),
        eval_every=100000,
        wm=model,
        action_dim=env.action_low.size,
        objective_fn=_create_planning_objective(
            model, preprocessor, config, baseline
        ),
        preprocessor=preprocessor,
        evaluator=None,
        wandb_run=_NoopRun(),
        log_filename=None,
        action_lower=normalized_low,
        action_upper=normalized_high,
    )
    episodes = []
    planning_latencies = []
    horizon = int(config["env"]["episode_horizon"])
    for episode in range(int(episodes_count)):
        observation = env.reset(seed + 1000 + episode)
        planner.clear_goal_cache()
        frames = [observation.rgb.copy()] if episode == 0 and video_path else []
        frame_metrics = [{"reward": 0.0, "success": False}] if frames else []
        goal_observation = _goal_observation(env_name, env, observation)
        episode_return = 0.0
        step_rewards: list[float] = []
        step_actions: list[list[float]] = []
        success = False
        warm_start = None
        for step in range(horizon):
            current = {
                "visual": observation.rgb[None, None],
                "proprio": observation.control_state[None, None],
            }
            started = time.perf_counter()
            with torch.no_grad():
                normalized_actions, _ = planner.plan(
                    current, goal_observation, actions=warm_start
                )
            planning_latencies.append(time.perf_counter() - started)
            warm_start = normalized_actions[:, 1:].detach()
            native = preprocessor.denormalize_actions(normalized_actions[0, 0]).cpu().numpy()
            native = np.clip(native, env.action_low, env.action_high)
            step_actions.append(native.astype(float).tolist())
            result = env.step(native)
            observation = result.observation
            episode_return += float(result.reward)
            step_rewards.append(float(result.reward))
            success = success or bool(result.info.get("success", False))
            if frames:
                frames.append(observation.rgb.copy())
                frame_metrics.append(
                    {"reward": float(result.reward), "success": bool(success)}
                )
            if result.done:
                break
        achieved = np.asarray(env.get_achieved_goal(), np.float32)
        desired = np.asarray(env.get_goal(), np.float32)
        episodes.append(
            {
                "seed": seed + 1000 + episode,
                "return": episode_return,
                "rewards": step_rewards,
                "actions": step_actions,
                "success": success,
                "episode_length": step + 1,
                "planning_latency": float(np.mean(planning_latencies[-(step + 1) :])),
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
                "DINO-WM",
                env_name,
                seed + 1000 + episode,
                horizon,
            )
    env.close()
    return episodes, planning_latencies


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    import torch

    source, source_commit = _load_official_source(
        request["config"]["baseline"]["source_path"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Official DINO-WM ViT predictor requires CUDA in this release")
    seed = int(request["seed"])
    seed_everything(seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    output.mkdir(parents=True, exist_ok=True)
    baseline = request["config"]["baseline"]
    model_settings = _model_settings(baseline)
    transitions = _load_transitions(
        request["offline_data"],
        int(request["offline_transitions"]),
        model_settings["history_frames"],
    )
    action_std = np.maximum(transitions.actions.std(axis=0), 0.05).astype(np.float32)
    proprio_all = np.concatenate(
        [transitions.current_proprio, transitions.next_proprio], axis=0
    )
    proprio_std = np.maximum(proprio_all.std(axis=0), 1e-4).astype(np.float32)
    preprocessor = DinoPreprocessor(
        transitions.actions.mean(axis=0).astype(np.float32),
        action_std,
        proprio_all.mean(axis=0).astype(np.float32),
        proprio_std,
    )
    model = _build_model(
        transitions.current_proprio.shape[-1],
        transitions.actions.shape[-1],
        device,
        model_settings,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(request["config"]["baseline"].get("learning_rate", 5e-4))
    )
    rng = np.random.default_rng(seed)
    losses = []
    train_started = time.perf_counter()
    model.train()
    explicit_updates = baseline.get("gradient_updates")
    epochs = int(baseline.get("epochs", 5))
    if epochs <= 0:
        raise ValueError("baseline.epochs must be positive")
    batch_size = min(
        int(request["config"]["baseline"].get("batch_size", 4)),
        len(transitions.window_starts),
    )
    if explicit_updates is None:
        update_batches = []
        for _ in range(epochs):
            order = rng.permutation(transitions.window_starts)
            update_batches.extend(
                order[start : start + batch_size]
                for start in range(0, len(order), batch_size)
            )
    else:
        count = int(explicit_updates)
        if count <= 0:
            raise ValueError("baseline.gradient_updates must be positive when set")
        update_batches = [
            rng.choice(
                transitions.window_starts, size=batch_size, replace=False
            )
            for _ in range(count)
        ]
    structured_losses = []
    projection = model.proprio_encoder.patch_embed
    projection_inverse = torch.linalg.pinv(
        projection.weight[..., 0].detach()
    )
    for indices in update_batches:
        observation, actions = _batch(transitions, indices, preprocessor, device)
        predicted_tokens, _, _, loss, loss_components = model(observation, actions)
        proprio_loss_weight = float(baseline.get("proprio_loss_weight", 0.0))
        if proprio_loss_weight:
            loss = loss + proprio_loss_weight * loss_components["z_proprio_loss"]
        predicted_normalized_proprio = (
            predicted_tokens[:, :, -2] - projection.bias
        ) @ projection_inverse.T
        structured_loss = torch.nn.functional.mse_loss(
            predicted_normalized_proprio,
            observation["proprio"][:, 1:],
        )
        loss = loss + float(
            baseline.get("structured_state_loss_weight", 1.0)
        ) * structured_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable, float(baseline.get("grad_clip", 1.0))
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        structured_losses.append(float(structured_loss.detach()))
    updates = len(update_batches)
    train_seconds = time.perf_counter() - train_started

    checkpoint_directory = Path(request.get("checkpoint_directory", output / "checkpoint"))
    checkpoint = checkpoint_directory / "final_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "proprio_encoder": model.proprio_encoder.state_dict(),
            "action_encoder": model.action_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_settings": model_settings,
            "preprocessor": {
                "action_mean": preprocessor.action_mean,
                "action_std": preprocessor.action_std,
                "proprio_mean": preprocessor.proprio_mean,
                "proprio_std": preprocessor.proprio_std,
            },
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location=device)
    reloaded = _build_model(
        transitions.current_proprio.shape[-1],
        transitions.actions.shape[-1],
        device,
        payload["model_settings"],
    )
    for name in ["encoder", "predictor", "proprio_encoder", "action_encoder"]:
        getattr(reloaded, name).load_state_dict(payload[name])
    reloaded.eval()
    stored_preprocessor = payload["preprocessor"]
    reloaded_preprocessor = DinoPreprocessor(
        stored_preprocessor["action_mean"].cpu().numpy(),
        stored_preprocessor["action_std"].cpu().numpy(),
        stored_preprocessor["proprio_mean"].cpu().numpy(),
        stored_preprocessor["proprio_std"].cpu().numpy(),
    )
    episodes, planning_latencies = _evaluate(
        reloaded,
        reloaded_preprocessor,
        request["config"],
        seed,
        request["config"]["baseline"],
        int(request["config"].get("baseline_eval_episodes", 5)),
        output / "videos" / "eval.gif",
    )
    write_episode_metrics(output / "metrics.jsonl", episodes)
    summary = {
        "status": "complete",
        "algorithm": "DINO-WM",
        "implementation": (
            "audited official DINOv2/VWorldModel with offline structured-state "
            "and bounded-CEM adaptation"
        ),
        "official_source": str(source),
        "official_commit": source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_reloaded": True,
        "environment_steps": 0,
        "training_environment_steps": 0,
        "offline_transitions": int(request["offline_transitions"]),
        "offline_dataset": str(Path(request["offline_data"]).resolve()),
        "gradient_updates": updates,
        "training_epochs": epochs if explicit_updates is None else None,
        "last_loss": losses[-1],
        "structured_state_loss_first": float(
            np.mean(structured_losses[: min(20, len(structured_losses))])
        ),
        "structured_state_loss_last": float(
            np.mean(structured_losses[-min(20, len(structured_losses)) :])
        ),
        "proprio_loss_weight": float(baseline.get("proprio_loss_weight", 0.0)),
        "structured_state_loss_weight": float(
            baseline.get("structured_state_loss_weight", 1.0)
        ),
        "loss_first": float(np.mean(losses[: min(20, len(losses))])),
        "loss_last": float(np.mean(losses[-min(20, len(losses)) :])),
        "wall_clock_train_seconds": train_seconds,
        "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "device": str(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "evaluation_metrics_file": "metrics.jsonl",
        "per_step_rewards": True,
        "observation_mode": "DINOv2 RGB patches+control_state; goal is target-only",
        "world_model_goal_conditioning": False,
        "planner": "bounded official-CEM update",
        "planning_objective": str(
            resolve_task_setting(
                baseline,
                str(request["config"]["env"]["name"]),
                "planning_objective",
                "latent",
            )
        ),
        "planner_warm_start": True,
        "planner_action_bounds_enforced": True,
        "planner_rgb_encoding_reused_across_candidates": True,
        "model_settings": model_settings,
        "planning_history_frames": int(
            resolve_task_setting(
                baseline,
                str(request["config"]["env"]["name"]),
                "planning_history_frames",
                model_settings["history_frames"],
            )
        ),
        "training_windows": int(len(transitions.window_starts)),
        "planning_latency_mean": float(np.mean(planning_latencies)),
        "legacy_mujoco_imported": False,
        **summarize_episodes(episodes, bootstrap_samples=1000),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    print(json.dumps(run(request, Path(args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
