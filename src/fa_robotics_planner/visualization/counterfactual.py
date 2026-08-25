from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_counterfactual(
    passive_frames: list[np.ndarray],
    adapted_frames: list[np.ndarray],
    simulator_frames: list[np.ndarray],
    output: str | Path,
) -> None:
    count = min(len(passive_frames), len(adapted_frames), len(simulator_frames))
    if count == 0:
        raise ValueError("Counterfactual visualization needs at least one frame")
    rows = []
    labels = ("State Prior", "State Prior + State Adapter", "Simulator")
    for timestep in range(count):
        images = [Image.fromarray(frames[timestep]).convert("RGB") for frames in (passive_frames, adapted_frames, simulator_frames)]
        width, height = images[0].size
        canvas = Image.new("RGB", (3 * width, height + 18), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(image, (index * width, 18))
            draw.text((index * width + 2, 2), label, fill="black")
        rows.append(np.asarray(canvas))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(target, np.asarray(rows), fps=8)

