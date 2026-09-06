from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_counterfactual(
    passive_frames: list[np.ndarray],
    adapted_frames: list[np.ndarray],
    simulator_frames: list[np.ndarray],
    output: str | Path,
    *,
    scale: int = 1,
) -> None:
    if int(scale) < 1:
        raise ValueError("Counterfactual visualization scale must be at least 1")
    count = min(len(passive_frames), len(adapted_frames), len(simulator_frames))
    if count == 0:
        raise ValueError("Counterfactual visualization needs at least one frame")
    rows = []
    labels = ("State Prior", "State Prior + State Adapter", "Simulator")
    for timestep in range(count):
        images = [
            Image.fromarray(frames[timestep]).convert("RGB").resize(
                (
                    int(frames[timestep].shape[1]) * int(scale),
                    int(frames[timestep].shape[0]) * int(scale),
                ),
                resample=Image.Resampling.NEAREST,
            )
            for frames in (passive_frames, adapted_frames, simulator_frames)
        ]
        width, height = images[0].size
        canvas = Image.new("RGB", (3 * width, height + 18), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (image, label) in enumerate(zip(images, labels)):
            canvas.paste(image, (index * width, 18))
            draw.text((index * width + 2, 2), label, fill="black")
        # Besides making long rollouts easier to inspect, an explicit timestep
        # prevents GIF encoders from collapsing consecutive static frames.
        timestep_label = f"t={timestep}"
        draw.text((3 * width - 6 * len(timestep_label), 2), timestep_label, fill="black")
        rows.append(np.asarray(canvas))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(target, np.asarray(rows), fps=8)
