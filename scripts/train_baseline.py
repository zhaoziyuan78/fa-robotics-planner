from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fa_robotics_planner.baselines import ExternalBaselineRunner, evaluate_gcrl, train_gcrl
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.evaluation.metrics import summarize_episodes
from fa_robotics_planner.experiments.run import RunDirectory

from ._common import checkpoint_path, config_from_unknown


def main() -> None:
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    baseline_name = next(
        (item.split("=", 1)[1] for item in unknown if item.startswith("baseline=")),
        None,
    )
    if baseline_name is None:
        raise ValueError("Specify baseline=gcrl|dreamerv3|tdmpc2|dino_wm")
    config = config_from_unknown(unknown)
    baseline = config["baseline"]
    env_name = config["env"]["name"]
    seed = int(config.get("seed", 0))
    steps = int(config.get("environment_steps", config.get("profile", {}).get("transitions", 1000)))
    experiment_id = str(config.get("experiment_id", f"{env_name}_{baseline_name}_seed{seed}"))
    checkpoint_directory = checkpoint_path(config, "baselines", experiment_id)
    run = RunDirectory(config.get("run_root", "runs"), experiment_id)
    run.initialize(
        config,
        {
            "seed": seed,
            "env_id": env_name,
            "method": baseline["algorithm"],
            "state_adapter": False,
            "action_adapter": False,
            "observation_mode": baseline.get(
                "observation", "control_state+goal"
            ),
            "paired_steps": steps
            if baseline_name == "dino_wm"
            else int(config.get("paired_steps", 0)),
            "state_only_steps": 0,
            "action_only_steps": 0,
            "planner_horizon": int(config.get("planner", {}).get("horizon", 0)),
            "num_candidates": int(config.get("planner", {}).get("num_candidates", 0)),
            "ood_config": {},
            "trainable_parameters": 0,
        },
    )
    if baseline_name == "gcrl":
        env = make_env(config)
        task_uses_her = env_name in set(baseline.get("her_tasks", []))
        baseline_config = {
            **baseline,
            "use_her": task_uses_her,
            "episode_horizon": config["env"].get("episode_horizon", 50),
        }
        started = time.perf_counter()
        checkpoint = train_gcrl(
            env,
            baseline_config,
            run.path,
            steps,
            seed,
            checkpoint_directory=checkpoint_directory,
        )
        train_seconds = time.perf_counter() - started
        evaluation = evaluate_gcrl(
            checkpoint,
            env,
            int(config["env"].get("episode_horizon", 50)),
            [seed + 1000 + index for index in range(int(config.get("baseline_eval_episodes", 5)))],
            use_goal_reward=task_uses_her,
            video_path=run.path / "videos" / "eval.gif",
        )
        for episode in evaluation:
            run.append_metric(episode)
        summary = {
            "status": "complete",
            "checkpoint": str(checkpoint),
            "environment_steps": steps,
            "wall_clock_train_seconds": train_seconds,
            **summarize_episodes(evaluation, bootstrap_samples=1000),
        }
        run.write_json("summary.json", summary)
        print(json.dumps(summary, indent=2))
    else:
        runner = ExternalBaselineRunner(
            baseline_name,
            baseline.get("external_command"),
            baseline.get("workdir"),
        )
        status = runner.run(
            {
                "baseline": baseline_name,
                "environment": env_name,
                "environment_steps": steps,
                "seed": seed,
                "config": config,
                "checkpoint_directory": str(checkpoint_directory),
            },
            run.path,
        )
        if status.status != "complete":
            run.write_json("summary.json", {"status": status.status, "reason": status.reason})
        print(json.dumps(status.__dict__, indent=2))
        if status.status == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
