from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fa_robotics_planner.config import compose_config
from fa_robotics_planner.data.generate import generate_dataset, generate_episode
from fa_robotics_planner.data.schemas import DatasetKind
from fa_robotics_planner.data.writer import EpisodeWriter
from fa_robotics_planner.envs import make_env

from ._common import ROOT, checkpoint_path, config_from_unknown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(unknown)
    env_name = config["env"]["name"]
    dataset = DatasetKind(config.get("dataset", "paired"))
    output = Path(args.output or config.get("output", config.get("data_root", "data")))
    if args.output is None and "output" not in config:
        output = output / env_name / dataset.value
    episodes = int(config.get("episodes", config.get("profile", {}).get("episodes", 5)))
    horizon = int(config.get("horizon", config["env"].get("episode_horizon", 50)))
    seed = int(config.get("seed", 0))
    metadata = {
        "environment": env_name,
        "seed": seed,
        "state_only_frames": horizon * episodes if dataset is DatasetKind.STATE_ONLY else 0,
        "action_only_steps": horizon * episodes if dataset is DatasetKind.ACTION_ONLY else 0,
        "paired_transitions": horizon * episodes if dataset is DatasetKind.PAIRED else 0,
    }
    if env_name == "humanoid_shared":
        if dataset is DatasetKind.PAIRED:
            raise ValueError("Paired Humanoid data is task-specific; choose one humanoid_<task> env")
        nominal = (
            config.get("nominal_controller")
            or config.get("nominal_controller_checkpoint")
            or config["env"].get("nominal_controller")
            or config.get("humanoid_nominal_controller", {}).get("checkpoint")
            or checkpoint_path(config, "infrastructure", "h1hand_stand_ppo.zip")
        )
        if not nominal:
            raise ValueError("humanoid_shared generation requires nominal_controller=<checkpoint>")
        tasks = ("humanoid_stand", "humanoid_balance", "humanoid_reach", "humanoid_push")
        envs = []
        for task in tasks:
            task_config = compose_config([f"env={task}"], config_root=ROOT / "configs")
            task_config["env"]["nominal_controller"] = nominal
            envs.append(make_env(task_config))
        writer = EpisodeWriter(output, dataset, metadata)
        probabilities = np.asarray([config["env"]["task_sampling"][name] for name in ("stand", "balance", "reach", "push")])
        rng = np.random.default_rng(seed)
        for episode in range(episodes):
            env = envs[int(rng.choice(len(envs), p=probabilities / probabilities.sum()))]
            writer.write(
                episode,
                generate_episode(
                    env,
                    dataset,
                    seed + episode,
                    horizon,
                    paired_config=config.get(
                        "paired_data", config["env"].get("paired_data")
                    ),
                ),
            )
        for env in envs:
            env.close()
    else:
        env = make_env(config)
        generate_dataset(
            env,
            str(output),
            dataset,
            episodes,
            horizon,
            seed,
            metadata,
            paired_config=config.get(
                "paired_data", config["env"].get("paired_data")
            ),
        )
        env.close()
    print(f"Wrote {dataset.value} dataset to {output}")


if __name__ == "__main__":
    main()
