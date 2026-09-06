import json
from pathlib import Path

import numpy as np

from scripts.plot_reward_curve import (
    aggregate_seed_curves,
    collect_seed_curves,
    plot_reward_curves,
    select_seed_curves,
    write_curve_csv,
)


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    rewards: list[list[float]],
    method: str = "FunctionAlignmentWM",
    state_adapter: bool = True,
    action_adapter: bool = True,
) -> None:
    run = root / name
    run.mkdir()
    metadata = {
        "created_at": f"2026-01-{seed + 1:02d}T00:00:00+00:00",
        "env_id": "windy",
        "eval": "id",
        "method": method,
        "seed": seed,
        "state_adapter": state_adapter,
        "action_adapter": action_adapter,
        "paired_steps": 10,
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = {}
    if method != "FunctionAlignmentWM":
        summary.update(
            training_environment_steps=0,
            offline_transitions=10,
        )
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    baseline_config = ""
    if method != "FunctionAlignmentWM":
        baseline_config = "baseline:\n  offline_only: true\n"
    (run / "config.yaml").write_text(
        "env:\n  episode_horizon: 2\ncondition: id\n" + baseline_config,
        encoding="utf-8",
    )
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(rewards):
            handle.write(
                json.dumps(
                    {
                        "seed": 100 + index,
                        "return": sum(sequence),
                        "rewards": sequence,
                    }
                )
                + "\n"
            )


def test_reward_curve_averages_episodes_then_seeds_equally(tmp_path):
    _write_run(tmp_path, "full_seed0", seed=0, rewards=[[1, 0], [3, 0]])
    _write_run(tmp_path, "full_seed1", seed=1, rewards=[[4, 2]])
    _write_run(
        tmp_path,
        "priors_seed0",
        seed=0,
        rewards=[[0, 1]],
        state_adapter=False,
        action_adapter=False,
    )
    _write_run(
        tmp_path,
        "baseline_seed0",
        seed=0,
        rewards=[[0.5, 0.5]],
        method="TT",
        state_adapter=False,
        action_adapter=False,
    )

    seeds, notes = collect_seed_curves(tmp_path, "windy")
    assert notes == []
    curves = aggregate_seed_curves(seeds)
    by_name = {curve.label: curve for curve in curves}
    # Seed 0 mean is [2, 2], seed 1 is [4, 6]. Seeds receive equal weight.
    np.testing.assert_allclose(by_name["Function Alignment"].mean, [3, 4])
    assert by_name["Function Alignment"].seeds == 2
    assert by_name["Function Alignment"].episodes == 3
    np.testing.assert_allclose(by_name["Priors only"].mean, [0, 1])
    np.testing.assert_allclose(by_name["TT"].mean, [0.5, 1.0])

    image = plot_reward_curves(
        curves, tmp_path / "reward.png", "windy", "cumulative"
    )
    csv_path = write_curve_csv(curves, tmp_path / "reward.csv")
    assert image.stat().st_size > 0
    assert csv_path.read_text(encoding="utf-8").startswith("method,family,step")


def test_reward_curve_reports_legacy_runs_without_step_rewards(tmp_path):
    _write_run(tmp_path, "legacy", seed=0, rewards=[[1, 0]])
    metrics = tmp_path / "legacy" / "metrics.jsonl"
    metrics.write_text(json.dumps({"seed": 100, "return": 1.0}) + "\n")

    curves, notes = collect_seed_curves(tmp_path, "windy")
    assert curves == []
    assert "rerun evaluation" in notes[0]


def test_reward_curve_reads_legacy_baseline_metrics_without_double_counting(tmp_path):
    _write_run(
        tmp_path,
        "baseline_seed0",
        seed=0,
        rewards=[[1.0, 1.0]],
        method="DINO-WM",
        state_adapter=False,
        action_adapter=False,
    )
    run = tmp_path / "baseline_seed0"
    legacy = run / "baseline_eval_metrics.jsonl"
    (run / "metrics.jsonl").replace(legacy)

    curves, notes = collect_seed_curves(tmp_path, "windy")
    assert notes == []
    assert len(curves) == 1
    assert curves[0].label == "DINO-WM"
    assert curves[0].episodes == 1

    # A rerun writes the canonical file but can leave the legacy file behind.
    # Prefer it instead of counting both versions of the same evaluation.
    (run / "metrics.jsonl").write_text(
        json.dumps({"seed": 101, "return": 4.0, "rewards": [2.0, 2.0]})
        + "\n",
        encoding="utf-8",
    )
    curves, notes = collect_seed_curves(tmp_path, "windy")
    assert notes == []
    assert curves[0].episodes == 1
    np.testing.assert_allclose(curves[0].values, [2.0, 4.0])


def test_reward_curve_imports_only_baselines_from_additional_roots(tmp_path):
    primary = tmp_path / "paper"
    legacy = tmp_path / "legacy"
    primary.mkdir()
    legacy.mkdir()
    _write_run(primary, "full_seed0", seed=0, rewards=[[1.0, 0.0]])
    _write_run(
        legacy,
        "tt_seed0",
        seed=0,
        rewards=[[0.5, 0.5]],
        method="TT",
        state_adapter=False,
        action_adapter=False,
    )
    # Additional roots may contain nested paper runs too; they must not replace
    # the authoritative main/ablation records from --runs.
    _write_run(
        legacy,
        "unrelated_full_seed1",
        seed=1,
        rewards=[[9.0, 9.0]],
    )

    seeds, notes = collect_seed_curves(
        primary, "windy", baseline_runs_roots=[legacy]
    )
    assert notes == []
    curves = aggregate_seed_curves(seeds)
    assert [curve.label for curve in curves] == [
        "Function Alignment",
        "TT",
    ]
    assert curves[0].seeds == 1
    assert curves[1].seeds == 1


def test_reward_curve_can_exclude_baselines(tmp_path):
    _write_run(tmp_path, "full_seed0", seed=0, rewards=[[1.0, 0.0]])
    _write_run(
        tmp_path,
        "baseline_seed0",
        seed=0,
        rewards=[[0.5, 0.5]],
        method="GCRL",
        state_adapter=False,
        action_adapter=False,
    )

    seeds, notes = collect_seed_curves(tmp_path, "windy")
    assert notes == []
    without_baselines = select_seed_curves(seeds, include_baselines=False)
    with_baselines = select_seed_curves(seeds, include_baselines=True)

    assert [curve.label for curve in aggregate_seed_curves(without_baselines)] == [
        "Function Alignment"
    ]
    assert [curve.label for curve in aggregate_seed_curves(with_baselines)] == [
        "Function Alignment",
        "GCRL",
    ]
