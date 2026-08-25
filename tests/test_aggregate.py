import json

from fa_robotics_planner.experiments.aggregate import aggregate


def test_aggregate_accepts_bootstrap_metrics_and_skips_failed_runs(tmp_path):
    runs = tmp_path / "runs"
    complete = runs / "complete"
    failed = runs / "failed"
    complete.mkdir(parents=True)
    failed.mkdir(parents=True)
    (complete / "metadata.json").write_text(
        json.dumps({"env_id": "windy", "method": "test", "seed": 0})
    )
    (complete / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "success": {"mean": 0.5, "low": 0.1, "high": 0.9},
                "return": {"mean": 2.0, "low": 1.0, "high": 3.0},
            }
        )
    )
    (failed / "metadata.json").write_text(json.dumps({"env_id": "windy"}))
    (failed / "summary.json").write_text(
        json.dumps({"status": "failed", "reason": "expected test failure"})
    )

    rows = aggregate(runs, tmp_path / "results")

    assert len(rows) == 1
    assert rows[0]["success_rate"] == 0.5
    assert rows[0]["episode_return"] == 2.0
    assert "status=failed" in (tmp_path / "results" / "missing_runs.md").read_text()
