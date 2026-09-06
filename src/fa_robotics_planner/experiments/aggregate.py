"""Aggregate run summaries to CSV, JSON, LaTeX, plots, and missing report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEYS = (
    "experiment_id",
    "env_id",
    "method",
    "seed",
    "eval",
    "state_adapter",
    "action_adapter",
    "success_rate",
    "episode_return",
    "paired_steps",
)
REMOVED_BASELINES = {"DreamerV3", "TD-MPC2", "GC-SAC"}
ACTIVE_BASELINES = {"GCRL", "TT", "DINO-WM"}


def _mean_scalar(value: Any, default: float = float("nan")) -> float:
    """Normalize scalar metrics and bootstrap summary objects."""

    while isinstance(value, dict):
        if "mean" not in value:
            return default
        value = value["mean"]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate(runs: str | Path, output: str | Path) -> list[dict[str, Any]]:
    runs, output = Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in sorted(path for path in runs.glob("*") if path.is_dir()):
        metadata_path, summary_path = run / "metadata.json", run / "summary.json"
        if not metadata_path.exists() or not summary_path.exists():
            missing.append(f"- `{run.name}`: missing metadata.json or summary.json")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("method") in REMOVED_BASELINES:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status", "complete") != "complete":
            missing.append(
                f"- `{run.name}`: status={summary.get('status', 'unknown')}"
            )
            continue
        if metadata.get("method") in ACTIVE_BASELINES and (
            int(summary.get("training_environment_steps", -1)) != 0
            or "offline_transitions" not in summary
        ):
            missing.append(
                f"- `{run.name}`: baseline predates the shared offline protocol"
            )
            continue
        success = summary.get("success", summary.get("success_rate", {}))
        returns = summary.get("return", summary.get("episode_return", {}))
        rows.append(
            {
                **metadata,
                "experiment_id": run.name,
                "success_rate": _mean_scalar(success),
                "episode_return": _mean_scalar(returns),
            }
        )
    (output / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(KEYS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    latex = ["\\begin{tabular}{llllrr}", "Environment & Method & Eval & Seed & Success & Return \\\\", "\\hline"]
    for row in rows:
        latex.append(
            f"{row.get('env_id','')} & {row.get('method','')} & {row.get('eval','')} & "
            f"{row.get('seed','')} & {float(row.get('success_rate', float('nan'))):.3f} & "
            f"{float(row.get('episode_return', float('nan'))):.3f} \\\\"
        )
    latex.append("\\end{tabular}")
    (output / "results.tex").write_text("\n".join(latex), encoding="utf-8")
    (output / "missing_runs.md").write_text("# Missing runs\n\n" + ("\n".join(missing) if missing else "None.\n"), encoding="utf-8")
    from fa_robotics_planner.visualization.plots import generate_standard_plots

    generate_standard_plots(rows, output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    rows = aggregate(args.runs, args.output)
    print(f"Aggregated {len(rows)} complete runs into {args.output}")


if __name__ == "__main__":
    main()
