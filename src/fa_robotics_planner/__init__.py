"""FunctionAlignmentWM robotics planning package."""

import os
from pathlib import Path

_cache_root = Path("/tmp/fa-robotics-planner-cache")
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

from .models.method import FunctionAlignmentWM

__all__ = ["FunctionAlignmentWM"]
__version__ = "0.1.0"
