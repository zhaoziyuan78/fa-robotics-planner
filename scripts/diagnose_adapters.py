"""Report held-out one-step errors for a trained prior/adapter checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.models.distributions import TanhNormal

from ._common import config_from_unknown
from .train_adapters import _load_priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--state-prior", required=True)
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--adapters", required=True)
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", *unknown])
    dataset = LazyEpisodeDataset(args.data, split="val", seed=int(config.get("seed", 0)))
    sample = dataset[0]
    config["env"]["proprio_size"] = int(sample["proprio"].shape[-1])
    method = build_method(config)
    _load_priors(method, Path(args.state_prior), Path(args.action_prior))
    checkpoint = torch.load(args.adapters, map_location="cpu", weights_only=False)
    method.state_adapter.load_state_dict(checkpoint["state_adapter"])
    method.action_adapter.load_state_dict(checkpoint["action_adapter"])
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    method.to(device).eval()

    squared = {name: [] for name in ("passive", "adapted", "intervention", "action")}
    action_scale = []
    with torch.inference_mode():
        for episode in dataset:
            state = torch.as_tensor(episode["control_state"], device=device)
            mask = torch.as_tensor(episode["state_mask"], device=device)
            next_state = torch.as_tensor(episode["next_control_state"], device=device)
            action = torch.as_tensor(episode["actions"], device=device)
            goal = torch.as_tensor(episode["goals"], device=device)
            proprio = torch.as_tensor(episode["proprio"], device=device)
            rgb = torch.as_tensor(episode["rgb"], device=device)
            valid = torch.ones(1, state.size(0), dtype=torch.bool, device=device)
            visual = method.visual_encoder(rgb.unsqueeze(0)) if method.visual_encoder else None
            prior = method.state_prior(
                state.unsqueeze(0), mask.unsqueeze(0), proprio.unsqueeze(0), visual, valid
            )
            passive = prior.passive_next.squeeze(0)
            adapted, _ = method.state_adapter(
                state, passive, action, prior.hidden.squeeze(0)
            )
            squared["passive"].append((passive - next_state).square())
            squared["adapted"].append((adapted - next_state).square())
            squared["intervention"].append(
                ((adapted - passive) - (next_state - passive)).square()
            )

            action_prior = method.action_prior(action[:-1].unsqueeze(0))
            base = TanhNormal(
                action_prior.distribution.loc.squeeze(0),
                action_prior.distribution.log_scale.squeeze(0),
                action_prior.distribution.low,
                action_prior.distribution.high,
            )
            proposal, _, _ = method.action_adapter(
                base, action_prior.hidden.squeeze(0), state, goal
            )
            expert = torch.as_tensor(
                episode.get("action_is_expert", torch.ones(state.size(0), dtype=bool)),
                dtype=torch.bool,
                device=device,
            )
            if expert.any():
                squared["action"].append((proposal.mean[expert] - action[expert]).square())
                action_scale.append(proposal.log_scale.exp()[expert])

    for name, chunks in squared.items():
        error = torch.cat(chunks).mean(0).sqrt().cpu()
        print(f"{name}_rmse={error.tolist()}")
    scales = torch.cat(action_scale).mean(0).cpu()
    print(f"action_raw_scale_mean={scales.tolist()}")


if __name__ == "__main__":
    main()
