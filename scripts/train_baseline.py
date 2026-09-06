from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fa_robotics_planner.baselines import ExternalBaselineRunner, evaluate_gcrl, train_gcrl
from fa_robotics_planner.baselines.env_adapter import resolve_task_setting
from fa_robotics_planner.baselines.offline_data import resolve_paired_data_path
from fa_robotics_planner.baselines.trajectory_transformer import (
    evaluate_trajectory_transformer,
    train_trajectory_transformer,
)
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.evaluation.metrics import (
    summarize_episodes,
    write_episode_metrics,
)
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
        raise ValueError("Specify baseline=gcrl|tt|dino_wm")
    if baseline_name not in {"gcrl", "tt", "dino_wm"}:
        raise ValueError(
            f"Unsupported baseline {baseline_name!r}; available offline baselines are "
            "gcrl, tt, and dino_wm"
        )
    config = config_from_unknown(unknown)
    baseline = config["baseline"]
    env_name = config["env"]["name"]
    seed = int(config.get("seed", 0))
    offline_transitions = int(
        config.get(
            "offline_transitions",
            # Compatibility for old launch files: this value now caps static
            # replay data and never authorizes online collection.
            config.get(
                "environment_steps", config.get("profile", {}).get("transitions", 1000)
            ),
        )
    )
    offline_data = resolve_paired_data_path(
        config.get("data_root", "data"), env_name, baseline.get("data")
    )
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
            "paired_steps": offline_transitions,
            "state_only_steps": 0,
            "action_only_steps": 0,
            "planner_horizon": int(
                baseline.get(
                    "planning_horizon",
                    config.get("planner", {}).get("horizon", 0),
                )
            ),
            "num_candidates": int(
                baseline.get(
                    "beam_width",
                    config.get("planner", {}).get("num_candidates", 0),
                )
            ),
            "ood_config": {},
            "trainable_parameters": 0,
            "eval": "id",
            "training_environment_steps": 0,
            "offline_dataset": str(offline_data),
        },
    )

    def update_metadata(**updates) -> None:
        path = run.path / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(updates)
        run.write_json("metadata.json", metadata)

    if baseline_name in {"gcrl", "tt"}:
        goal_reward_type = str(
            resolve_task_setting(
                baseline, env_name, "goal_reward_type", "dense"
            )
        )
        baseline_config = {
            **baseline,
            "device": config.get("device", "cuda"),
            "episode_horizon": config["env"].get("episode_horizon", 50),
            "goal_reward_type": goal_reward_type,
            "success_reward_threshold": (
                config["env"].get("paired_data", {}).get(
                    "success_reward_threshold"
                )
                if bool(config["env"].get("tt_truncate_on_success", False))
                else None
            ),
        }
        started = time.perf_counter()
        if baseline_name == "gcrl":
            checkpoint, diagnostics = train_gcrl(
                offline_data,
                config["env"],
                baseline_config,
                run.path,
                offline_transitions,
                seed,
                checkpoint_directory=checkpoint_directory,
            )
            evaluate = evaluate_gcrl
        else:
            checkpoint, diagnostics = train_trajectory_transformer(
                offline_data,
                baseline_config,
                run.path,
                offline_transitions,
                seed,
                checkpoint_directory=checkpoint_directory,
            )
            evaluate = evaluate_trajectory_transformer
        train_seconds = time.perf_counter() - started
        # Environment construction is intentionally delayed until every
        # optimizer update has completed.
        env = make_env(config)
        evaluation = evaluate(
            checkpoint,
            env,
            int(config["env"].get("episode_horizon", 50)),
            [seed + 1000 + index for index in range(int(config.get("baseline_eval_episodes", 5)))],
            video_path=run.path / "videos" / "eval.gif",
        )
        env.close()
        write_episode_metrics(run.path / "metrics.jsonl", evaluation)
        summary = {
            "status": "complete",
            "algorithm": baseline["algorithm"],
            "checkpoint": str(checkpoint),
            "environment_steps": 0,
            "training_environment_steps": 0,
            "offline_transitions": offline_transitions,
            "offline_dataset": str(offline_data),
            "wall_clock_train_seconds": train_seconds,
            "evaluation_metrics_file": "metrics.jsonl",
            "per_step_rewards": True,
            "training_reward": goal_reward_type if baseline_name == "gcrl" else "native_return_conditioned",
            **diagnostics,
            **summarize_episodes(evaluation, bootstrap_samples=1000),
        }
        update_metadata(
            trainable_parameters=int(diagnostics["parameter_count"]),
            training_environment_steps=0,
        )
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
                "environment_steps": 0,
                "offline_transitions": offline_transitions,
                "offline_data": str(offline_data),
                "seed": seed,
                "config": config,
                "checkpoint_directory": str(checkpoint_directory),
            },
            run.path,
        )
        if status.status != "complete":
            run.write_json("summary.json", {"status": status.status, "reason": status.reason})
        else:
            external_summary = json.loads(
                (run.path / "summary.json").read_text(encoding="utf-8")
            )
            update_metadata(
                trainable_parameters=int(external_summary.get("trainable_parameters", 0)),
                training_environment_steps=0,
            )
        print(json.dumps(status.__dict__, indent=2))
        if status.status == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
