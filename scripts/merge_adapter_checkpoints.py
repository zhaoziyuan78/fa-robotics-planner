"""Combine independently trained State and Action Adapter checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-source", required=True)
    parser.add_argument("--action-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state_source = Path(args.state_source)
    action_source = Path(args.action_source)
    state_checkpoint = torch.load(
        state_source, map_location="cpu", weights_only=False
    )
    action_checkpoint = torch.load(
        action_source, map_location="cpu", weights_only=False
    )
    if state_checkpoint.get("state_adapter") is None:
        raise ValueError(f"No State Adapter in {state_source}")
    if action_checkpoint.get("action_adapter") is None:
        raise ValueError(f"No Action Adapter in {action_source}")
    merged = dict(state_checkpoint)
    merged["kind"] = "Full"
    merged["action_adapter"] = action_checkpoint["action_adapter"]
    merged["config"] = action_checkpoint.get(
        "config", state_checkpoint.get("config", {})
    )
    model_config = merged["config"].setdefault("model", {})
    model_config.setdefault("state_adapter", {})["enabled"] = True
    model_config.setdefault("action_adapter", {})["enabled"] = True
    merged["component_sources"] = {
        "state_adapter": str(state_source.resolve()),
        "action_adapter": str(action_source.resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"Saved merged adapters to {output}")


if __name__ == "__main__":
    main()
