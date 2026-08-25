"""Train the one shared H1 standing infrastructure controller."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import humanoid_bench  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ._common import checkpoint_path, config_from_unknown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(unknown)
    steps = (
        args.steps
        if args.steps is not None
        else int(config.get("nominal_controller_steps", 1_000_000))
    )
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    device = args.device or str(config.get("device", "auto"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = Path(
        args.output
        or config.get(
            "nominal_controller_checkpoint",
            checkpoint_path(config, "infrastructure", "h1hand_stand_ppo.zip"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("h1hand-stand-v0", render_mode="rgb_array")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000, steps // 10),
        save_path=str(output.parent / "intermediate"),
        name_prefix="h1hand_stand",
    )
    started = time.perf_counter()
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        n_steps=min(2048, max(64, steps)),
        batch_size=64,
        verbose=1,
    )
    model.learn(total_timesteps=steps, callback=checkpoint_callback)
    model.save(output.with_suffix(""))
    elapsed = time.perf_counter() - started
    metadata = {
        "environment": "h1hand-stand-v0",
        "algorithm": "PPO",
        "seed": seed,
        "environment_steps": steps,
        "wall_clock_seconds": elapsed,
        "shared_infrastructure": True,
        "checkpoint": str(output),
    }
    (output.parent / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
