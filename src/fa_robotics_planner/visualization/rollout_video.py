from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import imageio.v3 as iio
import numpy as np


def save_rollout_video(
    frames: Iterable[np.ndarray],
    metrics: Iterable[Mapping[str, Any]],
    output: str | Path,
    method: str,
    task: str,
    seed: int,
    horizon: int,
    ood_parameters: Mapping[str, Any] | None = None,
    fps: int = 10,
) -> None:
    # Evaluation videos are evidence of the environment trajectory. Text is
    # stored in metrics/summary files instead of covering the rendered frame.
    rendered = [np.asarray(frame, np.uint8) for frame in frames]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(target, np.asarray(rendered), fps=fps)
