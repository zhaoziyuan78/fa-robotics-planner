"""Collect compact on-policy paired data with expert action relabelling."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import trange

from fa_robotics_planner.data.generate import generate_relabelled_episode
from fa_robotics_planner.data.schemas import DatasetKind
from fa_robotics_planner.data.writer import EpisodeWriter
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.utils import seed_everything

from ._common import config_from_unknown
from .evaluate import MethodPolicy
from .train_adapters import _load_priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-prior", required=True)
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--output", required=True)
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", "planner=shooting", *unknown])
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    env = make_env(config)
    method = build_method(config, env.action_low, env.action_high)
    _load_priors(method, Path(args.state_prior), Path(args.action_prior))
    checkpoint = torch.load(args.adapters, map_location="cpu", weights_only=False)
    if checkpoint.get("state_adapter") is not None:
        method.state_adapter.load_state_dict(checkpoint["state_adapter"])
    if checkpoint.get("action_adapter") is not None:
        method.action_adapter.load_state_dict(checkpoint["action_adapter"])
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    method.to(device).eval()
    policy = MethodPolicy(method, env, config, device)
    episodes = int(config.get("episodes", 200))
    horizon = int(config.get("horizon", config["env"].get("episode_horizon", 50)))
    writer = EpisodeWriter(
        args.output,
        DatasetKind.PAIRED,
        {
            "environment": config["env"]["name"],
            "seed": seed,
            "paired_transitions": 0,
            "collection": "on_policy_expert_relabelled",
            "source_adapters": str(Path(args.adapters).resolve()),
        },
    )
    for episode in trange(episodes, desc="relabel", unit="episode"):
        writer.write(
            episode,
            generate_relabelled_episode(
                env,
                policy,
                seed + episode,
                horizon,
                stop_on_success=bool(config.get("stop_on_success", True)),
            ),
        )
    env.close()
    print(f"Wrote relabelled paired dataset to {args.output}")


if __name__ == "__main__":
    main()
