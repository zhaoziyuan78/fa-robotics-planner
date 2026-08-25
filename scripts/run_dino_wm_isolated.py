"""Launch a legacy DINO-WM worker inside a separately managed Conda env."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--environment", default="dino-wm")
    parser.add_argument("--entrypoint", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.environment,
        "python",
        args.entrypoint,
        "--request",
        args.request,
        "--output",
        args.output,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "command": command,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
