"""Run HumanoidBench's bundled DreamerV3 through the baseline JSON protocol."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from fa_robotics_planner.baselines.env_adapter import (
    FlatObservationEnvAdapter,
    baseline_observation_metadata,
)
from fa_robotics_planner.envs import make_env as make_unified_env
from fa_robotics_planner.evaluation.metrics import summarize_episodes
from fa_robotics_planner.utils.seed import seed_everything


def _config(request: dict[str, Any], output: Path):
    import embodied
    import jax
    from embodied.agents.dreamerv3 import agent as dreamer_agent

    config = embodied.Config(dreamer_agent.Agent.configs["defaults"])
    baseline = request["config"]["baseline"]
    model_size = str(baseline.get("model_size", "small"))
    if model_size not in {"small", "medium", "large", "xlarge", "debug"}:
        raise ValueError(f"Unsupported DreamerV3 model_size: {model_size}")
    config = config.update(dreamer_agent.Agent.configs[model_size])
    config = config.update(dreamer_agent.Agent.configs["humanoid_proprio"])
    requested_platform = str(baseline.get("jax_platform", "auto"))
    if requested_platform == "auto":
        requested_platform = (
            "gpu" if any(device.platform == "gpu" for device in jax.devices()) else "cpu"
        )
    return config.update(
        {
            "task": f"humanoid_FA-{request['environment']}-v0",
            "method": "dreamerv3",
            "logdir": str(output),
            "seed": int(request["seed"]),
            "replay_size": max(1000, int(request["environment_steps"]) + 1),
            "batch_size": int(baseline.get("batch_size", 4)),
            "batch_length": int(baseline.get("batch_length", 8)),
            "encoder.mlp_keys": "vector",
            "encoder.cnn_keys": "$^",
            "decoder.mlp_keys": "vector",
            "decoder.cnn_keys": "$^",
            "wrapper.length": int(request["config"]["env"]["episode_horizon"]),
            "wrapper.checks": True,
            "run.steps": int(request["environment_steps"]),
            "run.num_envs": 1,
            "run.driver_parallel": False,
            "run.train_ratio": float(baseline.get("train_ratio", 4.0)),
            "run.train_fill": int(baseline.get("train_fill", 8)),
            "run.log_every": 100000,
            "run.save_every": 0,
            "run.log_video_streams": 0,
            "run.usage.nvsmi": False,
            "jax.platform": requested_platform,
            "jax.prealloc": False,
            "jax.transfer_guard": False,
        }
    )


def _make_env(
    project_config: dict[str, Any],
    dreamer_config,
    seed: int,
    index: int = 0,
    video_path: str | None = None,
):
    from embodied.agents.dreamerv3 import train as dreamer_train
    from embodied.envs.from_gymnasium import FromGymnasium

    unified = make_unified_env(project_config)
    gym_env = FlatObservationEnvAdapter(
        unified,
        int(project_config["env"]["episode_horizon"]),
        initial_seed=int(seed) + 10000 * int(index),
        record_video_path=video_path,
        record_method="DreamerV3",
        record_task=str(project_config["env"]["name"]),
    )
    return dreamer_train.wrap_env(
        FromGymnasium(gym_env, obs_key="vector"), dreamer_config
    )


def _parameter_count(agent) -> int:
    import jax

    return int(
        sum(
            int(getattr(leaf, "size", 0))
            for leaf in jax.tree_util.tree_leaves(agent.save())
        )
    )


def _evaluate(agent, make_env, episodes: int, seed: int) -> list[dict[str, Any]]:
    import embodied

    results: list[dict[str, Any]] = []
    current = {"return": 0.0, "length": 0, "success": False}
    driver = embodied.Driver([make_env], parallel=False)

    def record(transition, worker):
        del worker
        nonlocal current
        if transition["is_first"]:
            current = {"return": 0.0, "length": 0, "success": False}
        else:
            current["return"] += float(transition["reward"])
            current["length"] += 1
            current["success"] = current["success"] or bool(transition["success"])
        if transition["is_last"]:
            results.append(
                {
                    "seed": int(seed + len(results)),
                    "return": current["return"],
                    "success": current["success"],
                    "episode_length": current["length"],
                }
            )

    driver.on_step(record)
    driver.reset(agent.init_policy)
    policy = lambda *args: agent.policy(*args, mode="eval")
    driver(policy, episodes=int(episodes))
    driver.close()
    return results


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    import embodied
    import jax
    from embodied.agents.dreamerv3 import agent as dreamer_agent
    from embodied.agents.dreamerv3 import train as dreamer_train

    seed = int(request["seed"])
    seed_everything(seed)
    output.mkdir(parents=True, exist_ok=True)
    config = _config(request, output)
    project_config = request["config"]
    trained: dict[str, Any] = {}

    def make_training_env(index=0):
        return _make_env(project_config, config, seed, index)

    def make_agent():
        env = make_training_env(0)
        agent = dreamer_agent.Agent(env.obs_space, env.act_space, config)
        env.close()
        trained["agent"] = agent
        return agent

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    config.save(embodied.Path(output) / "resolved_official_config.yaml")
    started = time.perf_counter()
    embodied.run.train(
        make_agent,
        partial(dreamer_train.make_replay, config, "replay"),
        make_training_env,
        partial(dreamer_train.make_logger, config),
        args,
    )
    train_seconds = time.perf_counter() - started

    checkpoint_directory = Path(request.get("checkpoint_directory", output / "checkpoint"))
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = embodied.Path(checkpoint_directory) / "final_model.ckpt"
    final_saver = embodied.Checkpoint(checkpoint, parallel=False)
    final_saver.agent = trained["agent"]
    final_saver.save(keys=["agent"])
    eval_env = _make_env(project_config, config, seed + 1000)
    reloaded = dreamer_agent.Agent(eval_env.obs_space, eval_env.act_space, config)
    eval_env.close()
    # The bundled helper initializes its `_promise` member only in parallel
    # mode but reads it unconditionally from load(). Use that supported path.
    loader = embodied.Checkpoint(checkpoint, parallel=True)
    loader.agent = reloaded
    loader.load(keys=["agent"])
    eval_seed = seed + 1000
    episodes = _evaluate(
        reloaded,
        lambda: _make_env(
            project_config,
            config,
            eval_seed,
            video_path=str(output / "videos" / "eval.gif"),
        ),
        episodes=int(project_config.get("baseline_eval_episodes", 5)),
        seed=eval_seed,
    )
    with (output / "baseline_eval_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")
    summary = {
        "status": "complete",
        "algorithm": "DreamerV3",
        "implementation": "HumanoidBench bundled DreamerV3",
        "checkpoint": str(checkpoint),
        "checkpoint_reloaded": True,
        "environment_steps": int(request["environment_steps"]),
        "train_ratio": float(config.run.train_ratio),
        "model_size": str(project_config["baseline"].get("model_size", "small")),
        "batch_size": int(config.batch_size),
        "batch_length": int(config.batch_length),
        "recurrent_state_reset": "is_first at every environment reset",
        "parameter_count": _parameter_count(reloaded),
        "device": str(jax.devices()[0]),
        "peak_gpu_memory_bytes": 0,
        "wall_clock_train_seconds": train_seconds,
        "protocol_notes": [
            "JAX uses the available platform selected at runtime",
            "public state observation is used consistently across vector baselines",
        ],
        **baseline_observation_metadata(),
        **summarize_episodes(episodes, bootstrap_samples=1000),
    }
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
