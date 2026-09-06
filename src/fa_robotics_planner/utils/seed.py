"""Deterministic seeding without importing optional frameworks eagerly."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, env: Any | None = None, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    # JAX has no mutable global random state: callers must construct keys from
    # this seed themselves. Importing it here cannot seed later computations
    # and needlessly loads the JAX CUDA runtime in otherwise PyTorch processes.
    if env is not None:
        if hasattr(env, "reset"):
            env.reset(seed=seed)
        for space_name in ("action_space", "observation_space"):
            space = getattr(env, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(seed)
