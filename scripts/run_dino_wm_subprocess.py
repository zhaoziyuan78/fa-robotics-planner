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

from fa_robotics_planner.baselines.env_adapter import flatten_observation
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.envs.unified import ObservationBundle
from fa_robotics_planner.evaluation.metrics import summarize_episodes
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

    def transform_obs(self, observation: dict[str, np.ndarray]):
        import torch
        import torch.nn.functional as functional

        visual = torch.as_tensor(observation["visual"], dtype=torch.float32)
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
        proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32)
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        return {"visual": visual, "proprio": proprio}


@dataclass
class OfflineTransitions:
    current_rgb: np.ndarray
    current_proprio: np.ndarray
    actions: np.ndarray
    next_rgb: np.ndarray
    next_proprio: np.ndarray


def _collect_transitions(config: dict[str, Any], steps: int, seed: int) -> OfflineTransitions:
    env = make_env(config)
    rng = np.random.default_rng(seed)
    observation = env.reset(seed)
    current_rgb, current_proprio, actions = [], [], []
    next_rgb, next_proprio = [], []
    smooth_action = np.zeros_like(env.action_low)
    episode = 0
    for _ in range(int(steps)):
        innovation = rng.uniform(-1.0, 1.0, env.action_low.shape).astype(np.float32)
        smooth_action = np.clip(0.75 * smooth_action + 0.25 * innovation, -1.0, 1.0)
        native_action = env.action_low + 0.5 * (smooth_action + 1.0) * (
            env.action_high - env.action_low
        )
        result = env.step(native_action)
        current_rgb.append(observation.rgb)
        current_proprio.append(flatten_observation(observation))
        actions.append(native_action.astype(np.float32))
        next_rgb.append(result.observation.rgb)
        next_proprio.append(flatten_observation(result.observation))
        observation = result.observation
        if result.done:
            episode += 1
            observation = env.reset(seed + episode)
    env.close()
    return OfflineTransitions(
        np.stack(current_rgb),
        np.stack(current_proprio),
        np.stack(actions),
        np.stack(next_rgb),
        np.stack(next_proprio),
    )


def _build_model(proprio_size: int, action_size: int, device):
    from models.dino import DinoV2Encoder
    from models.proprio import ProprioceptiveEmbedding
    from models.visual_world_model import VWorldModel
    from models.vit import ViTPredictor

    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    proprio_encoder = ProprioceptiveEmbedding(
        num_frames=1, in_chans=int(proprio_size), emb_dim=encoder.emb_dim
    )
    action_encoder = ProprioceptiveEmbedding(
        num_frames=1, in_chans=int(action_size), emb_dim=encoder.emb_dim
    )
    predictor = ViTPredictor(
        num_patches=198,
        num_frames=1,
        dim=encoder.emb_dim,
        depth=1,
        heads=4,
        mlp_dim=512,
        dropout=0.0,
        emb_dropout=0.0,
        pool="mean",
    )
    model = VWorldModel(
        image_size=224,
        num_hist=1,
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
    return model.to(device)


def _batch(
    transitions: OfflineTransitions,
    indices: np.ndarray,
    preprocessor: DinoPreprocessor,
    device,
):
    import torch

    observation = {
        "visual": np.stack(
            [transitions.current_rgb[indices], transitions.next_rgb[indices]], axis=1
        ),
        "proprio": np.stack(
            [
                transitions.current_proprio[indices],
                transitions.next_proprio[indices],
            ],
            axis=1,
        ),
    }
    observation = {
        name: value.to(device) for name, value in preprocessor.transform_obs(observation).items()
    }
    actions = np.stack(
        [transitions.actions[indices], np.zeros_like(transitions.actions[indices])],
        axis=1,
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
        "proprio": flatten_observation(target)[None, None],
    }


class _NoopRun:
    def log(self, *args, **kwargs):
        del args, kwargs


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
    from planning.cem import CEMPlanner
    from planning.objectives import create_objective_fn

    env_name = str(config["env"]["name"])
    if env_name.startswith("humanoid_"):
        raise NotImplementedError(
            "DINO-WM strategy-A smoke is validated for Windy and Fetch; Humanoid needs a goal-image protocol"
        )
    env = make_env(config)
    planner = CEMPlanner(
        horizon=int(baseline.get("horizon", 2)),
        topk=int(baseline.get("topk", 2)),
        num_samples=int(baseline.get("num_samples", 4)),
        var_scale=float(baseline.get("var_scale", 0.5)),
        opt_steps=int(baseline.get("opt_steps", 1)),
        eval_every=100000,
        wm=model,
        action_dim=env.action_low.size,
        objective_fn=create_objective_fn(alpha=float(baseline.get("proprio_alpha", 10.0)), base=1.0),
        preprocessor=preprocessor,
        evaluator=None,
        wandb_run=_NoopRun(),
        log_filename=None,
    )
    episodes = []
    planning_latencies = []
    horizon = int(config["env"]["episode_horizon"])
    for episode in range(int(episodes_count)):
        observation = env.reset(seed + 1000 + episode)
        frames = [observation.rgb.copy()] if episode == 0 and video_path else []
        frame_metrics = [{"reward": 0.0, "success": False}] if frames else []
        goal_observation = _goal_observation(env_name, env, observation)
        episode_return = 0.0
        success = False
        for step in range(horizon):
            current = {
                "visual": observation.rgb[None, None],
                "proprio": flatten_observation(observation)[None, None],
            }
            started = time.perf_counter()
            with torch.no_grad():
                normalized_actions, _ = planner.plan(current, goal_observation)
            planning_latencies.append(time.perf_counter() - started)
            native = preprocessor.denormalize_actions(normalized_actions[0, 0]).cpu().numpy()
            native = np.clip(native, env.action_low, env.action_high)
            result = env.step(native)
            observation = result.observation
            episode_return += float(result.reward)
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
    transitions = _collect_transitions(
        request["config"], int(request["environment_steps"]), seed
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
        transitions.current_proprio.shape[-1], transitions.actions.shape[-1], device
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(request["config"]["baseline"].get("learning_rate", 5e-4))
    )
    rng = np.random.default_rng(seed)
    losses = []
    train_started = time.perf_counter()
    model.train()
    baseline = request["config"]["baseline"]
    explicit_updates = baseline.get("gradient_updates")
    epochs = int(baseline.get("epochs", 5))
    if epochs <= 0:
        raise ValueError("baseline.epochs must be positive")
    batch_size = min(
        int(request["config"]["baseline"].get("batch_size", 4)),
        len(transitions.actions),
    )
    if explicit_updates is None:
        update_batches = []
        for _ in range(epochs):
            order = rng.permutation(len(transitions.actions))
            update_batches.extend(
                order[start : start + batch_size]
                for start in range(0, len(order), batch_size)
            )
    else:
        count = int(explicit_updates)
        if count <= 0:
            raise ValueError("baseline.gradient_updates must be positive when set")
        update_batches = [
            rng.choice(len(transitions.actions), size=batch_size, replace=False)
            for _ in range(count)
        ]
    for indices in update_batches:
        observation, actions = _batch(transitions, indices, preprocessor, device)
        _, _, _, loss, _ = model(observation, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
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
        transitions.current_proprio.shape[-1], transitions.actions.shape[-1], device
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
    with (output / "baseline_eval_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")
    summary = {
        "status": "complete",
        "algorithm": "DINO-WM",
        "implementation": "official strategy-A model and CEM over unified env",
        "official_source": str(source),
        "official_commit": source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_reloaded": True,
        "environment_steps": int(request["environment_steps"]),
        "gradient_updates": updates,
        "training_epochs": epochs if explicit_updates is None else None,
        "last_loss": losses[-1],
        "wall_clock_train_seconds": train_seconds,
        "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "device": str(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "observation_mode": "DINOv2 RGB patches+control_state+goal",
        "planner": "official CEM",
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
