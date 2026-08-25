from __future__ import annotations

import argparse

import numpy as np

from fa_robotics_planner.envs import make_env

from ._common import config_from_unknown, dump, resolve_env_override


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--seed", type=int, default=0)
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(resolve_env_override(args.env, unknown))
    env = make_env(config)
    first = env.reset(args.seed)
    first_state = first.control_state.copy()
    transition = env.step(np.zeros_like(env.action_low))
    repeated = env.reset(args.seed)
    if not np.array_equal(first_state, repeated.control_state):
        raise AssertionError("reset(seed) is not reproducible")
    if first.rgb.dtype != np.uint8 or first.rgb.ndim != 3 or first.rgb.shape[-1] != 3:
        raise AssertionError(f"Invalid RGB observation: {first.rgb.shape} {first.rgb.dtype}")
    if np.any(env.action_low >= env.action_high):
        raise AssertionError("Invalid action bounds")
    dump(
        {
            "env": config["env"]["name"],
            "rgb_shape": first.rgb.shape,
            "proprio_shape": first.proprio.shape,
            "control_state_shape": first.control_state.shape,
            "goal_shape": first.goal.shape,
            "action_shape": env.action_low.shape,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "success": transition.info.get("success", False),
            "status": "ok",
        }
    )
    env.close()


if __name__ == "__main__":
    main()

