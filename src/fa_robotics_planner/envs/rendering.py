"""Legacy-compatible WindyNav renderer."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


def world_to_pixel(position: np.ndarray, size: int) -> tuple[int, int]:
    x, y = position
    return int((x + 1) * 0.5 * (size - 1)), int((1 - (y + 1) * 0.5) * (size - 1))


def render_windy(
    size: int,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    region_colors: list[list[int]],
    grid_spacing: int = 4,
) -> np.ndarray:
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(reversed(region_colors)):
        y0 = int(index * size / len(region_colors))
        y1 = int((index + 1) * size / len(region_colors)) - 1
        draw.rectangle((0, y0, size - 1, y1), fill=tuple(color))
    for coordinate in range(0, size, max(1, grid_spacing)):
        draw.line((coordinate, 0, coordinate, size - 1), fill=(40, 40, 40))
        draw.line((0, coordinate, size - 1, coordinate), fill=(40, 40, 40))
    draw.rectangle((0, 0, size - 1, size - 1), outline=(170, 170, 170))
    gx, gy = world_to_pixel(goal, size)
    draw.line((gx - 3, gy, gx + 3, gy), fill=(90, 170, 120))
    draw.line((gx, gy - 3, gx, gy + 3), fill=(90, 170, 120))
    ax, ay = world_to_pixel(position, size)
    vx, vy = float(velocity[0]), -float(velocity[1])
    norm = max(math.hypot(vx, vy), 1e-6)
    vx, vy = vx / norm, vy / norm
    tip = (ax + vx * 4, ay + vy * 4)
    base = (ax - vx * 4, ay - vy * 4)
    px, py = -vy * 3, vx * 3
    draw.line((tip[0], tip[1], base[0] + px, base[1] + py), fill=(220, 120, 90), width=2)
    draw.line((tip[0], tip[1], base[0] - px, base[1] - py), fill=(220, 120, 90), width=2)
    return np.asarray(image, dtype=np.uint8)

