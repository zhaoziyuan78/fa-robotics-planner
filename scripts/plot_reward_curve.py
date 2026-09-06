"""Plot per-step evaluation reward for one environment across methods and seeds.

The evaluator stores one native-environment reward sequence per episode.  This
script first averages episodes within each training seed, then gives every seed
equal weight.  That two-level reduction prevents a run with more evaluation
episodes from dominating the paper curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fa_robotics_planner.config import load_yaml


METHOD_LABELS = {
    (True, True): ("Function Alignment", "main"),
    (True, False): ("State Adapter only", "ablation"),
    (False, True): ("Action Adapter only", "ablation"),
    (False, False): ("Priors only", "ablation"),
}
METHOD_ORDER = {
    "Function Alignment": 0,
    "State Adapter only": 1,
    "Action Adapter only": 2,
    "Priors only": 3,
    "GCRL": 10,
    "TT": 11,
    "DINO-WM": 12,
}
ACTIVE_BASELINE_LABELS = {"GCRL", "TT", "DINO-WM"}


@dataclass(frozen=True)
class SeedCurve:
    label: str
    family: str
    seed: int
    values: np.ndarray
    pad_value: float
    episodes: int
    run: Path
    created_at: str


@dataclass(frozen=True)
class MeanCurve:
    label: str
    family: str
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    seeds: int
    episodes: int


def _method_identity(
    metadata: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str]:
    if metadata.get("method") == "FunctionAlignmentWM":
        enabled = (
            bool(metadata.get("state_adapter", True)),
            bool(metadata.get("action_adapter", True)),
        )
        return METHOD_LABELS[enabled]
    baseline = config.get("baseline", {})
    label = str(
        metadata.get("method")
        or baseline.get("algorithm")
        or baseline.get("name")
        or "Baseline"
    )
    return label, "baseline"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_episode_metrics(run: Path) -> list[dict[str, Any]]:
    """Read canonical metrics, falling back to the legacy baseline filename.

    Never merge the two files: a run produced across a code upgrade can retain
    the old file, and counting both would duplicate its evaluation episodes.
    """

    paths = (run / "metrics.jsonl", run / "baseline_eval_metrics.jsonl")
    for path in paths:
        if not path.exists():
            continue
        episodes: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                if "rewards" in value:
                    episodes.append(value)
        if episodes:
            return episodes
    return []


def _run_curve(
    episodes: list[dict[str, Any]], horizon: int, metric: str
) -> np.ndarray:
    reward_sequences = [
        np.asarray(episode["rewards"], dtype=np.float64).reshape(-1)
        for episode in episodes
    ]
    if not reward_sequences or any(sequence.size == 0 for sequence in reward_sequences):
        raise ValueError("Reward sequences must be non-empty")
    horizon = max(int(horizon), max(sequence.size for sequence in reward_sequences))
    rows = []
    for rewards in reward_sequences:
        if not np.isfinite(rewards).all():
            raise ValueError("Reward sequence contains NaN or infinity")
        values = np.cumsum(rewards) if metric == "cumulative" else rewards
        pad_value = float(values[-1]) if metric == "cumulative" else 0.0
        rows.append(
            np.pad(
                values,
                (0, horizon - values.size),
                mode="constant",
                constant_values=pad_value,
            )
        )
    return np.stack(rows).mean(axis=0)


def collect_seed_curves(
    runs_root: str | Path,
    env: str,
    *,
    evaluation: str = "id",
    condition: str | None = None,
    metric: str = "cumulative",
    paired_steps: int | None = None,
    baseline_runs_roots: Iterable[str | Path] = (),
) -> tuple[list[SeedCurve], list[str]]:
    """Load the newest complete run for each method/training-seed pair.

    ``runs_root`` is authoritative for the main method and ablations. Optional
    baseline roots are searched only for baseline families, which lets a paper
    result directory include older baseline jobs written to the project-level
    ``runs/`` without accidentally importing unrelated main-method runs.
    """

    root = Path(runs_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    if evaluation == "ood" and not condition:
        raise ValueError("--condition is required when --eval=ood")
    if metric not in {"cumulative", "reward"}:
        raise ValueError("metric must be 'cumulative' or 'reward'")

    candidates: dict[tuple[str, int], SeedCurve] = {}
    notes: list[str] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        run = metadata_path.parent
        metadata = _read_json(metadata_path)
        if str(metadata.get("env_id", "")) != env:
            continue
        run_eval = str(metadata.get("eval", "id"))
        if run_eval != evaluation:
            continue
        summary_path = run / "summary.json"
        if not summary_path.exists():
            notes.append(f"skip {run.name}: missing summary.json")
            continue
        summary = _read_json(summary_path)
        if summary.get("status", "complete") != "complete":
            notes.append(
                f"skip {run.name}: status={summary.get('status', 'unknown')}"
            )
            continue
        config_path = run / "config.yaml"
        config = load_yaml(config_path) if config_path.exists() else {}
        run_condition = str(config.get("condition", "id"))
        if condition is not None and run_condition != condition:
            continue
        if paired_steps is not None and int(metadata.get("paired_steps", -1)) != int(
            paired_steps
        ):
            continue
        episodes = _read_episode_metrics(run)
        if not episodes:
            notes.append(
                f"skip {run.name}: no per-step rewards; rerun evaluation with the current code"
            )
            continue
        horizon = int(config.get("env", {}).get("episode_horizon", 0))
        label, family = _method_identity(metadata, config)
        if family == "baseline" and label not in ACTIVE_BASELINE_LABELS:
            notes.append(f"skip {run.name}: baseline {label} is no longer active")
            continue
        if family == "baseline" and (
            config.get("baseline", {}).get("offline_only") is not True
            or int(summary.get("training_environment_steps", -1)) != 0
            or "offline_transitions" not in summary
        ):
            notes.append(
                f"skip {run.name}: baseline result predates the shared offline protocol"
            )
            continue
        seed = int(metadata.get("seed", config.get("seed", 0)))
        values = _run_curve(episodes, horizon, metric)
        curve = SeedCurve(
            label=label,
            family=family,
            seed=seed,
            values=values,
            pad_value=float(values[-1]) if metric == "cumulative" else 0.0,
            episodes=len(episodes),
            run=run,
            created_at=str(
                metadata.get(
                    "created_at",
                    f"{metadata_path.stat().st_mtime_ns:020d}",
                )
            ),
        )
        key = (label, seed)
        previous = candidates.get(key)
        if previous is None or curve.created_at > previous.created_at:
            if previous is not None:
                notes.append(
                    f"use {run.name} instead of older duplicate {previous.run.name}"
                )
            candidates[key] = curve
        else:
            notes.append(f"skip older duplicate {run.name}")
    primary_root = root.resolve()
    for baseline_root_value in baseline_runs_roots:
        baseline_root = Path(baseline_root_value).expanduser()
        if baseline_root.resolve() == primary_root:
            continue
        baseline_curves, baseline_notes = collect_seed_curves(
            baseline_root,
            env,
            evaluation=evaluation,
            condition=condition,
            metric=metric,
            paired_steps=paired_steps,
        )
        notes.extend(
            f"baseline root {baseline_root}: {note}" for note in baseline_notes
        )
        for curve in baseline_curves:
            if curve.family != "baseline":
                continue
            key = (curve.label, curve.seed)
            previous = candidates.get(key)
            if previous is None or curve.created_at > previous.created_at:
                if previous is not None:
                    notes.append(
                        f"use {curve.run.name} instead of older duplicate "
                        f"{previous.run.name}"
                    )
                candidates[key] = curve
            else:
                notes.append(f"skip older duplicate {curve.run.name}")
    return list(candidates.values()), notes


def aggregate_seed_curves(seed_curves: list[SeedCurve]) -> list[MeanCurve]:
    grouped: dict[str, list[SeedCurve]] = {}
    for curve in seed_curves:
        grouped.setdefault(curve.label, []).append(curve)
    result = []
    for label, curves in grouped.items():
        horizon = max(curve.values.size for curve in curves)
        rows = []
        for curve in curves:
            rows.append(
                np.pad(
                    curve.values,
                    (0, horizon - curve.values.size),
                    mode="constant",
                    constant_values=curve.pad_value,
                )
            )
        matrix = np.stack(rows)
        mean = matrix.mean(axis=0)
        if matrix.shape[0] > 1:
            half_width = 1.96 * matrix.std(axis=0, ddof=1) / math.sqrt(
                matrix.shape[0]
            )
        else:
            half_width = np.zeros_like(mean)
        result.append(
            MeanCurve(
                label=label,
                family=curves[0].family,
                mean=mean,
                low=mean - half_width,
                high=mean + half_width,
                seeds=len(curves),
                episodes=sum(curve.episodes for curve in curves),
            )
        )
    return sorted(result, key=lambda item: (METHOD_ORDER.get(item.label, 100), item.label))


def select_seed_curves(
    seed_curves: Iterable[SeedCurve], *, include_baselines: bool
) -> list[SeedCurve]:
    """Apply the user-facing baseline visibility choice before aggregation."""

    return [
        curve
        for curve in seed_curves
        if include_baselines or curve.family != "baseline"
    ]


def plot_reward_curves(
    curves: list[MeanCurve],
    output: str | Path,
    env: str,
    metric: str,
    *,
    show_ci: bool = True,
    title: str | None = None,
) -> Path:
    if not curves:
        raise ValueError("No reward curves matched the requested filters")
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    family_style = {
        "main": {"linestyle": "-", "linewidth": 2.8},
        "ablation": {"linestyle": "--", "linewidth": 1.9},
        "baseline": {"linestyle": ":", "linewidth": 2.1},
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for index, curve in enumerate(curves):
        steps = np.arange(1, curve.mean.size + 1)
        color = colors[index % len(colors)]
        axis.plot(
            steps,
            curve.mean,
            color=color,
            label=f"{curve.label} ({curve.seeds} seeds)",
            **family_style[curve.family],
        )
        if show_ci and curve.seeds > 1:
            axis.fill_between(steps, curve.low, curve.high, color=color, alpha=0.16)
    axis.set_xlabel("Environment step")
    axis.set_ylabel(
        "Mean cumulative native reward"
        if metric == "cumulative"
        else "Mean native reward"
    )
    axis.set_title(title or f"{env}: reward by rollout step")
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def write_curve_csv(curves: list[MeanCurve], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "family",
                "step",
                "mean",
                "ci95_low",
                "ci95_high",
                "seeds",
                "episodes",
            ),
        )
        writer.writeheader()
        for curve in curves:
            for step, (mean, low, high) in enumerate(
                zip(curve.mean, curve.low, curve.high), 1
            ):
                writer.writerow(
                    {
                        "method": curve.label,
                        "family": curve.family,
                        "step": step,
                        "mean": float(mean),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "seeds": curve.seeds,
                        "episodes": curve.episodes,
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot main-method, adapter-ablation, and baseline reward curves for "
            "one environment."
        )
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--baseline-runs",
        action="append",
        default=[],
        help=(
            "Additional run root containing baseline experiments. May be "
            "specified more than once; main/ablation runs are still taken "
            "only from --runs."
        ),
    )
    baseline_visibility = parser.add_mutually_exclusive_group()
    baseline_visibility.add_argument(
        "--include-baselines",
        dest="include_baselines",
        action="store_true",
        help="Plot baseline methods (default).",
    )
    baseline_visibility.add_argument(
        "--exclude-baselines",
        dest="include_baselines",
        action="store_false",
        help="Plot only Function Alignment and its adapter ablations.",
    )
    parser.set_defaults(include_baselines=True)
    parser.add_argument("--out")
    parser.add_argument("--eval", choices=("id", "ood"), default="id")
    parser.add_argument("--condition")
    parser.add_argument(
        "--metric", choices=("cumulative", "reward"), default="cumulative"
    )
    parser.add_argument("--paired-steps", type=int)
    parser.add_argument("--title")
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    suffix = f"_{args.condition}" if args.condition else ""
    output = Path(
        args.out
        or f"results/reward_curve_{args.env}_{args.eval}{suffix}.png"
    )
    seed_curves, notes = collect_seed_curves(
        args.runs,
        args.env,
        evaluation=args.eval,
        condition=args.condition,
        metric=args.metric,
        paired_steps=args.paired_steps,
        baseline_runs_roots=(args.baseline_runs if args.include_baselines else ()),
    )
    seed_curves = select_seed_curves(
        seed_curves, include_baselines=args.include_baselines
    )
    curves = aggregate_seed_curves(seed_curves)
    for note in notes:
        print(f"warning: {note}")
    if not curves:
        raise SystemExit(
            "No per-step reward curves matched. Rerun the requested evaluations "
            "with the current evaluator and check --env/--eval/--condition."
        )
    plot_reward_curves(
        curves,
        output,
        args.env,
        args.metric,
        show_ci=not args.no_ci,
        title=args.title,
    )
    if not args.no_csv:
        write_curve_csv(curves, output.with_suffix(".csv"))
    print(
        f"Saved {len(curves)} methods from {len(seed_curves)} seeds to {output}"
    )


if __name__ == "__main__":
    main()
