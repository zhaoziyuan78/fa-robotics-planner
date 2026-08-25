from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fa_robotics_planner.config import save_config


class RunDirectory:
    def __init__(self, root: str | Path, experiment_id: str):
        self.path = Path(root) / experiment_id
        for directory in ("checkpoint", "videos", "plots"):
            (self.path / directory).mkdir(parents=True, exist_ok=True)
        for filename in ("stdout.log", "stderr.log"):
            (self.path / filename).touch(exist_ok=True)

    def initialize(self, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        save_config(config, self.path / "config.yaml")
        data = dict(metadata)
        data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        try:
            data.setdefault(
                "git_commit",
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip(),
            )
        except Exception:
            data.setdefault("git_commit", "")
        self.write_json("metadata.json", data)

    def append_metric(self, metric: Mapping[str, Any]) -> None:
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(metric), sort_keys=True) + "\n")

    def write_json(self, filename: str, value: Mapping[str, Any]) -> None:
        (self.path / filename).write_text(json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8")
