from __future__ import annotations

import argparse

from fa_robotics_planner.envs.ood import validate_ood_config

from ._common import config_from_unknown


def main() -> None:
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(unknown)
    errors = validate_ood_config(
        config["env"].get("train_ranges", {}),
        config["eval"].get("ood_conditions", {}),
        config.get("seeds", [0, 1, 2, 3, 4]),
        config["eval"].get("seeds", [100, 101, 102, 103, 104]),
    )
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        raise SystemExit(1)
    print("OOD configuration is valid")


if __name__ == "__main__":
    main()

