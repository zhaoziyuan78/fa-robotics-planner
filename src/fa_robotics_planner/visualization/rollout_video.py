from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    height = 12 * len(lines) + 6
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0))
    draw.multiline_text((3, 3), "\n".join(lines), fill=(255, 255, 255), spacing=1)
    return np.asarray(image)


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
    rendered = []
    for timestep, (frame, metric) in enumerate(zip(frames, metrics)):
        rendered.append(
            _overlay(
                frame,
                [
                    f"{method} | {task} | seed={seed} | t={timestep}",
                    f"reward={metric.get('reward', 0):.3f} success={metric.get('success', False)} H={horizon}",
                    f"OOD={dict(ood_parameters or {})}",
                ],
            )
        )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(target, np.asarray(rendered), fps=fps)

