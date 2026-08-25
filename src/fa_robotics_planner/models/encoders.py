from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VisualEncoder(nn.Module):
    """Small RGB encoder; high-resolution reconstruction is intentionally absent."""

    def __init__(self, output_dim: int = 128, normalize_output: bool = False):
        super().__init__()
        self.normalize_output = bool(normalize_output)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim < 3 or rgb.shape[-3] not in (3,) and rgb.shape[-1] not in (3,):
            raise ValueError("RGB input must end in HWC or CHW")
        leading = rgb.shape[:-3]
        if rgb.shape[-1] == 3:
            rgb = rgb.movedim(-1, -3)
        flat = rgb.reshape(-1, *rgb.shape[-3:]).float() / 255.0
        encoded = self.network(flat)
        if self.normalize_output:
            encoded = F.normalize(encoded, dim=-1, eps=1e-6)
        return encoded.reshape(*leading, -1)
