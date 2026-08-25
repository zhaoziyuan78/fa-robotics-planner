"""Trace one learned-policy episode against the paired-data controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fa_robotics_planner.data.generate import paired_expert_action
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.models.builders import build_method

from ._common import config_from_unknown
from .evaluate import MethodPolicy
from .train_adapters import _load_priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-prior", required=True)
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--execute-expert", action="store_true")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", "planner=shooting", *unknown])
    env = make_env(config)
    method = build_method(config, env.action_low, env.action_high)
    _load_priors(method, Path(args.state_prior), Path(args.action_prior))
    checkpoint = torch.load(args.adapters, map_location="cpu", weights_only=False)
    method.state_adapter.load_state_dict(checkpoint["state_adapter"])
    method.action_adapter.load_state_dict(checkpoint["action_adapter"])
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    method.to(device).eval()
    policy = MethodPolicy(method, env, config, device)
    observation = env.reset(args.episode_seed)
    trace = []
    for step in range(int(config["env"].get("episode_horizon", 50))):
        expert = paired_expert_action(env, observation)
        action, _ = policy(observation)
        executed = expert if args.execute_expert else action
        if args.execute_expert:
            # MethodPolicy has already advanced its causal action context with
            # the proposed action.  Replace that final token with the action
            # actually executed and rebuild the exact KV cache next step.
            expert_tensor = torch.as_tensor(expert, device=device).reshape(1, 1, -1)
            policy.history = torch.cat(
                (policy.history[:, :-1].clone(), expert_tensor), dim=1
            )
            policy._action_prior_output = None
        transition = env.step(executed)
        state = observation.control_state
        trace.append(
            {
                "step": step,
                "state": state.tolist(),
                "goal": observation.goal.tolist(),
                "policy_action": np.asarray(action).tolist(),
                "expert_action": np.asarray(expert).tolist(),
                "executed_action": np.asarray(executed).tolist(),
                "success": bool(transition.info.get("success", False)),
            }
        )
        observation = transition.observation
        if transition.done:
            break
    Path(args.output).write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"Wrote {len(trace)} traced steps to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
