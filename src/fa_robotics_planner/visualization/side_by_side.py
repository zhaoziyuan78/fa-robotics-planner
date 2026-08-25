from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def _read_frames(path: str | Path) -> list[np.ndarray]:
    frames = [np.asarray(frame, np.uint8)[..., :3] for frame in iio.imiter(path)]
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames


def save_side_by_side(
    left_video: str | Path,
    right_video: str | Path,
    output: str | Path,
    left_label: str = "Function Alignment",
    right_label: str = "Baseline",
    fps: int = 10,
) -> None:
    left = _read_frames(left_video)
    right = _read_frames(right_video)
    count = max(len(left), len(right))
    rendered: list[np.ndarray] = []
    for index in range(count):
        images = [
            Image.fromarray(left[min(index, len(left) - 1)]).convert("RGB"),
            Image.fromarray(right[min(index, len(right) - 1)]).convert("RGB"),
        ]
        height = max(image.height for image in images)
        resized = []
        for image in images:
            if image.height != height:
                width = round(image.width * height / image.height)
                image = image.resize((width, height))
            resized.append(image)
        canvas = Image.new(
            "RGB", (sum(image.width for image in resized), height + 20), "white"
        )
        draw = ImageDraw.Draw(canvas)
        offset = 0
        for image, label in zip(resized, (left_label, right_label)):
            canvas.paste(image, (offset, 20))
            draw.text((offset + 4, 3), label, fill="black")
            offset += image.width
        rendered.append(np.asarray(canvas))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(target, np.stack(rendered), fps=int(fps))
