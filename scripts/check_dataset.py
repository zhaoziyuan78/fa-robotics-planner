from __future__ import annotations

import argparse

from fa_robotics_planner.data import check_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--no-checksum", action="store_true")
    args = parser.parse_args()
    errors = check_dataset(args.root, not args.no_checksum)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"Dataset integrity OK: {args.root}")


if __name__ == "__main__":
    main()

