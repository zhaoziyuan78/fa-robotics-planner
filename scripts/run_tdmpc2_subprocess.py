"""Run the HumanoidBench-bundled official TD-MPC2 through the JSON protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fa_robotics_planner.baselines.env_adapter import (
    FlatObservationEnvAdapter,
    baseline_observation_metadata,
)
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.evaluation.metrics import summarize_episodes
from fa_robotics_planner.utils.seed import seed_everything


def _official_config(request: dict[str, Any], observation_size: int, action_size: int):
    from omegaconf import OmegaConf
    from tdmpc2.common import MODEL_SIZE

    import tdmpc2

    source = Path(tdmpc2.__file__).resolve().parent / "config.yaml"
    cfg = OmegaConf.load(source)
    baseline = request["config"]["baseline"]
    model_size = int(baseline.get("model_size", 1))
    if model_size not in MODEL_SIZE:
        raise ValueError(f"TD-MPC2 model_size must be one of {sorted(MODEL_SIZE)}")
    for name, value in MODEL_SIZE[model_size].items():
        cfg[name] = value

    cfg.task = f"fa-{request['environment']}"
    cfg.task_title = cfg.task
    cfg.tasks = [cfg.task]
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.model_size = model_size
    cfg.obs = "state"
    cfg.obs_shape = {"state": [int(observation_size)]}
    cfg.action_dim = int(action_size)
    cfg.episode_length = int(request["config"]["env"]["episode_horizon"])
    cfg.steps = int(request["environment_steps"])
    cfg.seed = int(request["seed"])
    cfg.seed_steps = min(cfg.episode_length, max(1, cfg.steps - 1))
    cfg.batch_size = int(baseline.get("batch_size", 8))
    cfg.horizon = int(baseline.get("horizon", 2))
    cfg.iterations = int(baseline.get("iterations", 2))
    cfg.num_samples = int(baseline.get("num_samples", 16))
    cfg.num_elites = min(int(baseline.get("num_elites", 4)), cfg.num_samples)
    cfg.num_pi_trajs = min(int(baseline.get("num_pi_trajs", 0)), cfg.num_samples)
    cfg.buffer_size = max(int(cfg.steps + 1), int(cfg.episode_length + 1))
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.disable_wandb = True
    cfg.save_video = False
    cfg.save_agent = True
    return cfg


def _transition(obs, action=None, reward=None):
    import torch
    from tensordict.tensordict import TensorDict

    obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full((1,), float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    return TensorDict(
        {"obs": obs, "action": action.unsqueeze(0), "reward": reward.unsqueeze(0)},
        batch_size=(1,),
    )


def _evaluate(agent, env, seeds: list[int], horizon: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        episode_return = 0.0
        success = False
        final_goal_distance = float("nan")
        for step in range(horizon):
            action = agent.act(obs, t0=step == 0, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            success = success or bool(info.get("success", False))
            final_goal_distance = float(info.get("goal_distance", final_goal_distance))
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": int(seed),
                "return": episode_return,
                "success": success,
                "episode_length": step + 1,
                "final_goal_distance": final_goal_distance,
            }
        )
    return episodes


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    import torch
    from tdmpc2.common.buffer import Buffer
    from tdmpc2.envs.wrappers.tensor import TensorWrapper
    from tdmpc2.tdmpc2 import TDMPC2

    class SeedableTensorWrapper(TensorWrapper):
        def reset(self, *, seed=None, options=None):
            observation, info = self.env.reset(seed=seed, options=options)
            return self._obs_to_tensor(observation), info

    seed = int(request["seed"])
    seed_everything(seed)
    output.mkdir(parents=True, exist_ok=True)
    unified = make_env(request["config"])
    gym_env = FlatObservationEnvAdapter(
        unified,
        int(request["config"]["env"]["episode_horizon"]),
        initial_seed=seed,
    )
    env = SeedableTensorWrapper(gym_env)
    first_obs, _ = env.reset(seed=seed)
    cfg = _official_config(request, first_obs.numel(), env.action_space.shape[0])
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)

    started = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    episode = [_transition(obs, torch.full_like(env.rand_act(), float("nan")))]
    gradient_updates = 0
    completed_episodes = 0
    train_metrics: dict[str, float] = {}
    for step in range(int(cfg.steps)):
        if step < cfg.seed_steps or buffer.num_eps == 0:
            action = env.rand_act()
        else:
            action = agent.act(obs, t0=len(episode) == 1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode.append(_transition(next_obs, action, reward))
        obs = next_obs
        if terminated or truncated:
            completed_episodes = buffer.add(torch.cat(episode))
            obs, _ = env.reset()
            episode = [_transition(obs, torch.full_like(env.rand_act(), float("nan")))]
        if buffer.num_eps and step >= cfg.seed_steps:
            train_metrics = agent.update(buffer)
            gradient_updates += 1

    train_seconds = time.perf_counter() - started
    checkpoint_directory = Path(request.get("checkpoint_directory", output / "checkpoint"))
    checkpoint = checkpoint_directory / "final_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint)
    parameter_count = int(agent.model.total_params)
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = TDMPC2(cfg)
    reloaded.load(checkpoint)
    eval_unified = make_env(request["config"])
    eval_gym = FlatObservationEnvAdapter(
        eval_unified,
        int(cfg.episode_length),
        initial_seed=seed + 1000,
        record_video_path=str(output / "videos" / "eval.gif"),
        record_method="TD-MPC2",
        record_task=str(request["environment"]),
    )
    eval_env = SeedableTensorWrapper(eval_gym)
    eval_seeds = [
        seed + 1000 + index
        for index in range(int(request["config"].get("baseline_eval_episodes", 5)))
    ]
    episodes = _evaluate(reloaded, eval_env, eval_seeds, int(cfg.episode_length))
    eval_env.close()
    for episode_result in episodes:
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_result, sort_keys=True) + "\n")
    summary = {
        "status": "complete",
        "algorithm": "TD-MPC2",
        "implementation": "HumanoidBench bundled official TD-MPC2",
        "checkpoint": str(checkpoint),
        "checkpoint_reloaded": True,
        "environment_steps": int(cfg.steps),
        "gradient_updates": gradient_updates,
        "completed_train_episodes": completed_episodes,
        "wall_clock_train_seconds": train_seconds,
        "parameter_count": parameter_count,
        "device": str(reloaded.device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "debug_protocol_deviations": [
            "one gradient update per post-seed environment step",
            "state observation used for the dependency smoke test",
        ],
        "last_train_metrics": train_metrics,
        **baseline_observation_metadata(),
        **summarize_episodes(episodes, bootstrap_samples=1000),
    }
    (output / "resolved_official_config.yaml").write_text(
        __import__("omegaconf").OmegaConf.to_yaml(cfg), encoding="utf-8"
    )
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
