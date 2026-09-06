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
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--state-prior", required=True)
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--adapters", required=True)
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", *unknown])
    dataset = LazyEpisodeDataset(args.data, split="val", seed=int(config.get("seed", 0)))
    token_dataset = LazyEpisodeDataset(args.tokens, split="val", seed=int(config.get("seed", 0)))
    if [entry["id"] for entry in dataset.entries] != [entry["id"] for entry in token_dataset.entries]:
        raise ValueError("Observation/token validation episode IDs differ")
    sample = dataset[0]
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
        for index, episode in enumerate(dataset):
            token_episode = token_dataset[index]
            state = torch.as_tensor(episode["control_state"], device=device)
            mask = torch.as_tensor(episode["state_mask"], device=device)
            next_state = torch.as_tensor(episode["next_control_state"], device=device)
            action = torch.as_tensor(episode["actions"], device=device)
            goal = torch.as_tensor(episode["goals"], device=device)
            video = torch.as_tensor(token_episode["video_tokens"], device=device)
            next_video = torch.as_tensor(token_episode["next_video_tokens"], device=device)
            valid = torch.ones(1, state.size(0) + 1, dtype=torch.bool, device=device)
            prior = method.state_prior(
                torch.cat((state, next_state[-1:]), 0).unsqueeze(0),
                torch.cat(
                    (
                        mask,
                        torch.as_tensor(
                            episode["next_state_mask"][-1:], device=device
                        ),
                    ),
                    0,
                ).unsqueeze(0),
                torch.cat((video, next_video[-1:]), 0).unsqueeze(0),
                valid,
            )
            passive = prior.passive_next.squeeze(0)
            adapted, _, _ = method.state_adapter(
                state,
                passive,
                action,
                prior.hidden.squeeze(0),
                prior.video_summary[:, :-1].squeeze(0),
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
