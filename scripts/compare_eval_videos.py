from __future__ import annotations

import argparse

from fa_robotics_planner.visualization import save_side_by_side


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Function Alignment and baseline evaluation GIFs."
    )
    parser.add_argument("--ours", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ours-label", default="Function Alignment")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    save_side_by_side(
        args.ours,
        args.baseline,
        args.output,
        args.ours_label,
        args.baseline_label,
        args.fps,
    )
    print(f"Saved comparison GIF to {args.output}")


if __name__ == "__main__":
    main()
