"""Auditable subprocess boundary for the official DINO-WM integration."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping


@dataclass
class BaselineStatus:
    baseline: str
    status: str
    command: list[str]
    returncode: int | None
    elapsed_seconds: float
    reason: str = ""


class ExternalBaselineRunner:
    def __init__(self, name: str, command: str | list[str] | None, workdir: str | Path | None = None):
        self.name = name
        self.command = shlex.split(command) if isinstance(command, str) else list(command or [])
        if self.command and self.command[0] in {"python", "python3"}:
            self.command[0] = sys.executable
        self.workdir = None if workdir is None else Path(workdir)

    def run(self, request: Mapping[str, Any], run_directory: str | Path) -> BaselineStatus:
        run = Path(run_directory)
        run.mkdir(parents=True, exist_ok=True)
        (run / "request.json").write_text(json.dumps(dict(request), indent=2, sort_keys=True), encoding="utf-8")
        if not self.command:
            status = BaselineStatus(
                self.name,
                "unsupported",
                [],
                None,
                0.0,
                "baseline.external_command is not configured; no algorithm was silently substituted",
            )
            (run / "baseline_status.json").write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
            return status
        started = time.perf_counter()
        with (run / "stdout.log").open("w", encoding="utf-8") as stdout, (run / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [*self.command, "--request", str(run / "request.json"), "--output", str(run)],
                cwd=self.workdir,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        elapsed = time.perf_counter() - started
        summary = run / "summary.json"
        ok = completed.returncode == 0 and summary.exists()
        status = BaselineStatus(
            self.name,
            "complete" if ok else "failed",
            self.command,
            completed.returncode,
            elapsed,
            "" if ok else "subprocess failed or did not emit summary.json; see stderr.log",
        )
        (run / "baseline_status.json").write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
        return status
