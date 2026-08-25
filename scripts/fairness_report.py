from __future__ import annotations

import argparse
import json
from pathlib import Path

from fa_robotics_planner.config import load_yaml
from fa_robotics_planner.experiments.fairness import fairness_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="results/fairness_report.md")
    args = parser.parse_args()
    methods = {}
    for run in sorted(Path(args.runs).glob("*")):
        if (
            not run.is_dir()
            or not (run / "config.yaml").exists()
            or not (run / "metadata.json").exists()
            or not (run / "summary.json").exists()
        ):
            continue
        config = load_yaml(run / "config.yaml")
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status", "complete") != "complete":
            continue
        env, planner = config.get("env", {}), config.get("planner", {})
        baseline = config.get("baseline", {})
        environment_steps = config.get("environment_steps", metadata.get("paired_steps", 0))
        paired_steps = metadata.get("paired_steps", 0)
        if baseline.get("name") == "dino_wm":
            paired_steps = environment_steps
        methods[run.name] = {
            "environment_steps": environment_steps,
            "paired_steps": paired_steps,
            "state_only_steps": metadata.get("state_only_steps", 0),
            "action_only_steps": metadata.get("action_only_steps", 0),
            "episode_horizon": env.get("episode_horizon"),
            "action_repeat": env.get("action_repeat", 1),
            "control_frequency": env.get("control_frequency", "environment default"),
            "observation_mode": summary.get(
                "observation_mode",
                baseline.get("observation", metadata.get("observation_mode")),
            ),
            "image_size": env.get("image_size"),
            "reward": env.get("reward", "environment native"),
            "goal_definition": env.get("goal_definition", "environment native"),
            "evaluation_seeds": config.get("eval", {}).get("seeds"),
            "planning_candidates": baseline.get("num_samples", planner.get("num_candidates", 0)),
            "planning_horizon": baseline.get("horizon", planner.get("horizon", 0)),
            "nominal_controller": env.get("nominal_controller"),
            "action_bounds": env.get("action_bounds", "environment native"),
        }
    if not methods:
        raise RuntimeError(f"No complete run configs found below {args.runs}")
    warnings = fairness_report(methods, args.output)
    print(f"Wrote {args.output} with {len(warnings)} mismatch warning(s)")


if __name__ == "__main__":
    main()
