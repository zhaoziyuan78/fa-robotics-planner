from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from fa_robotics_planner.config import save_config

from ._common import config_from_unknown


NAMES = {
    (True, True): "Full",
    (True, False): "StateAdapterOnly",
    (False, True): "ActionAdapterOnly",
    (False, False): "PriorsOnly",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", default="runs/ablation_configs")
    args, unknown = parser.parse_known_args()
    env_override = f"env={args.env}" if args.env else next((item for item in unknown if item.startswith("env=")), "env=windy")
    config = config_from_unknown(["model=prior_adapter", "planner=shooting", env_override, *unknown])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for state_adapter, action_adapter in itertools.product((True, False), repeat=2):
        for seed in map(int, args.seeds.split(",")):
            resolved = json.loads(json.dumps(config))
            resolved["seed"] = seed
            resolved["state_adapter"] = state_adapter
            resolved["action_adapter"] = action_adapter
            resolved["model"]["state_adapter"]["enabled"] = state_adapter
            resolved["model"]["action_adapter"]["enabled"] = action_adapter
            name = NAMES[(state_adapter, action_adapter)]
            run_id = f"{resolved['env']['name']}_{name}_seed{seed}"
            resolved["experiment_id"] = run_id
            path = output / f"{run_id}.yaml"
            save_config(resolved, path)
            manifest.append({"name": name, "seed": seed, "config": str(path)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} ablation run configs in {output}")


if __name__ == "__main__":
    main()

