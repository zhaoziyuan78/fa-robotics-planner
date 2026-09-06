"""Select strong paired-data episodes and export their stored RGB as GIFs.

This utility is deliberately dataset-only: it does not create an environment,
load a checkpoint, or run inference.  Selection is deterministic so the paper
visualizations can be regenerated from the recorded source episode.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import imageio.v3 as iio
import numpy as np

from fa_robotics_planner.config import load_yaml


TASKS = (
    "fetch_push",
    "fetch_slide",
    "humanoid_stand",
    "humanoid_balance",
    "humanoid_reach",
    "humanoid_push",
)
FETCH_TASKS = frozenset(("fetch_push", "fetch_slide"))
DEFAULT_VARIANTS = ("paper_v4", "paper_v2", None)


@dataclass(frozen=True)
class EpisodeSelection:
    task: str
    dataset: str
    episode_id: int
    episode_file: str
    length: int
    episode_return: float
    mean_reward: float
    success_steps: int | None
    final_success: bool | None
    minimum_valid_length: int
    selection_basis: str


def resolve_dataset(
    data_root: str | Path,
    task: str,
    *,
    override: str | Path | None = None,
) -> Path:
    """Resolve the newest available paper dataset, with explicit fallbacks."""

    if override is not None:
        candidates = [Path(override).expanduser()]
    else:
        root = Path(data_root).expanduser()
        candidates = [
            root / variant / task / "paired" if variant else root / task / "paired"
            for variant in DEFAULT_VARIANTS
        ]
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No paired dataset found for {task}; searched: {searched}")


def _read_manifest(dataset: Path) -> list[dict[str, Any]]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes listed in {manifest_path}")
    return episodes


def _candidate_score(
    task: str,
    *,
    episode_return: float,
    success_steps: int,
    final_success: bool,
    length: int,
    episode_id: int,
) -> tuple[float, ...]:
    # Fetch uses a sparse -1/0 reward. Prefer a trajectory which is successful
    # at the end and remains successful, rather than one that only crosses the
    # tolerance briefly. Humanoid rewards are dense, so return is the canonical
    # task-quality signal. The final terms make every tie deterministic.
    if task in FETCH_TASKS:
        return (
            float(final_success),
            float(success_steps),
            episode_return,
            float(length),
            float(-episode_id),
        )
    return (episode_return, float(length), float(-episode_id))


def select_best_episode(
    dataset: str | Path,
    task: str,
    *,
    minimum_length_fraction: float = 0.25,
) -> EpisodeSelection:
    """Choose one reproducible high-quality, non-degenerate episode."""

    if not 0.0 <= minimum_length_fraction <= 1.0:
        raise ValueError("minimum_length_fraction must be between 0 and 1")
    dataset = Path(dataset).expanduser()
    episodes = _read_manifest(dataset)
    longest = max(int(episode.get("length", 0)) for episode in episodes)
    minimum_length = max(2, int(math.ceil(longest * minimum_length_fraction)))
    best: tuple[tuple[float, ...], EpisodeSelection] | None = None
    for episode in episodes:
        episode_id = int(episode.get("id", -1))
        episode_file = str(episode.get("file", ""))
        if not episode_file:
            continue
        with np.load(dataset / episode_file, allow_pickle=False) as payload:
            rewards = np.asarray(payload["rewards"], dtype=np.float64).reshape(-1)
        length = int(rewards.size)
        if length < minimum_length or not np.isfinite(rewards).all():
            continue
        episode_return = float(rewards.sum())
        success_steps = int(np.count_nonzero(rewards >= 0.0))
        final_success = bool(rewards[-1] >= 0.0)
        score = _candidate_score(
            task,
            episode_return=episode_return,
            success_steps=success_steps,
            final_success=final_success,
            length=length,
            episode_id=episode_id,
        )
        selection = EpisodeSelection(
            task=task,
            dataset=str(dataset.resolve()),
            episode_id=episode_id,
            episode_file=episode_file,
            length=length,
            episode_return=episode_return,
            mean_reward=float(rewards.mean()),
            success_steps=success_steps if task in FETCH_TASKS else None,
            final_success=final_success if task in FETCH_TASKS else None,
            minimum_valid_length=minimum_length,
            selection_basis=(
                "final sparse success, sustained success, then return"
                if task in FETCH_TASKS
                else "maximum native episode return among non-degenerate episodes"
            ),
        )
        if best is None or score > best[0]:
            best = score, selection
    if best is None:
        raise ValueError(
            f"No finite episode of at least {minimum_length} steps in {dataset}"
        )
    return best[1]


def uniformly_spaced_indices(frame_count: int, max_frames: int) -> np.ndarray:
    """Keep both endpoints while bounding GIF size for long Humanoid runs."""

    if frame_count <= 0 or max_frames <= 0:
        raise ValueError("frame_count and max_frames must be positive")
    if frame_count <= max_frames:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, frame_count - 1, max_frames)).astype(np.int64)
    )


def load_episode_frames(selection: EpisodeSelection) -> np.ndarray:
    """Load observations plus the final next-observation from one episode."""

    source = Path(selection.dataset) / selection.episode_file
    with np.load(source, allow_pickle=False) as payload:
        frames = np.asarray(payload["rgb"], dtype=np.uint8)
        if "next_rgb" in payload and len(payload["next_rgb"]):
            final_frame = np.asarray(payload["next_rgb"][-1], dtype=np.uint8)
            frames = np.concatenate((frames, final_frame[None]), axis=0)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"Expected THWC RGB(A) frames in {source}, got {frames.shape}")
    return frames[..., :3]


def write_gif(
    frames: np.ndarray,
    output: str | Path,
    *,
    fps: float,
    max_frames: int,
) -> tuple[Path, np.ndarray]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    indices = uniformly_spaced_indices(len(frames), max_frames)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output,
        frames[indices],
        extension=".gif",
        loop=0,
        duration=1000.0 / fps,
    )
    return output, indices


def _parse_dataset_overrides(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or task not in TASKS or not path:
            raise ValueError(
                "--dataset must have the form TASK=PATH, where TASK is one of "
                + ", ".join(TASKS)
            )
        result[task] = Path(path).expanduser()
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_config = load_yaml(project_root / "configs" / "default.yaml")
    parser = argparse.ArgumentParser(
        description="Export one strong stored paired-data rollout per robotics task."
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Task to export; repeat as needed. Defaults to all Fetch/Humanoid tasks.",
    )
    parser.add_argument("--data-root", default=default_config.get("data_root", "data"))
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override one task's paired dataset path; may be repeated.",
    )
    parser.add_argument("--output", default="results/viz")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=160,
        help="Uniformly sample longer rollouts across their full duration.",
    )
    parser.add_argument(
        "--minimum-length-fraction",
        type=float,
        default=0.25,
        help="Ignore degenerate episodes shorter than this fraction of the dataset maximum.",
    )
    args = parser.parse_args()

    try:
        overrides = _parse_dataset_overrides(args.dataset)
    except ValueError as error:
        parser.error(str(error))
    tasks = tuple(args.task or TASKS)
    output_root = Path(args.output).expanduser()
    records: dict[str, dict[str, Any]] = {}
    for task in tasks:
        dataset = resolve_dataset(
            args.data_root,
            task,
            override=overrides.get(task),
        )
        selection = select_best_episode(
            dataset,
            task,
            minimum_length_fraction=args.minimum_length_fraction,
        )
        frames = load_episode_frames(selection)
        gif_path, frame_indices = write_gif(
            frames,
            output_root / f"{task}.gif",
            fps=args.fps,
            max_frames=args.max_frames,
        )
        record = asdict(selection)
        record.update(
            {
                "gif": str(gif_path.resolve()),
                "source_frames": int(len(frames)),
                "gif_frames": int(len(frame_indices)),
                "first_source_frame": int(frame_indices[0]),
                "last_source_frame": int(frame_indices[-1]),
                "fps": float(args.fps),
            }
        )
        records[task] = record
        print(
            f"{task}: episode {selection.episode_id}, "
            f"return={selection.episode_return:.3f}, "
            f"frames={len(frame_indices)}/{len(frames)} -> {gif_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "selection.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "selection_is_deterministic": True,
                "tasks": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Selection metadata -> {manifest_path}")


if __name__ == "__main__":
    main()
