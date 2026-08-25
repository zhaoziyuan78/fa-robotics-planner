"""Environment-neutral observation and transition contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _vector(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


@dataclass(frozen=True)
class ObservationBundle:
    rgb: np.ndarray
    proprio: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    control_state: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    state_mask: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    goal: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must have shape (H,W,3), got {rgb.shape}")
        proprio = _vector(self.proprio)
        control = _vector(self.control_state)
        mask = _vector(self.state_mask, np.bool_)
        goal = _vector(self.goal)
        if mask.shape != control.shape:
            raise ValueError("state_mask and control_state must have identical shape")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "control_state", control)
        object.__setattr__(self, "state_mask", mask)
        object.__setattr__(self, "goal", goal)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb,
            "proprio": self.proprio,
            "control_state": self.control_state,
            "state_mask": self.state_mask,
            "goal": self.goal,
        }


@dataclass(frozen=True)
class StepResult:
    observation: ObservationBundle
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class UnifiedControlEnv(ABC):
    """Minimal interface shared by the main method and every baseline."""

    @abstractmethod
    def reset(self, seed: int) -> ObservationBundle:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_goal(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_control_state(self) -> np.ndarray:
        raise NotImplementedError

    def get_state_mask(self) -> np.ndarray:
        return np.ones_like(self.get_control_state(), dtype=bool)

    @abstractmethod
    def compute_score(
        self,
        predicted_state: np.ndarray,
        goal: np.ndarray,
        action_sequence: np.ndarray | None = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_ood_parameters(self, config: Mapping[str, Any]) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


def pad_vector(value: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    value = _vector(value)
    if value.size > size:
        raise ValueError(f"Vector of size {value.size} exceeds configured padding {size}")
    padded = np.zeros(size, dtype=np.float32)
    mask = np.zeros(size, dtype=bool)
    padded[: value.size] = value
    mask[: value.size] = True
    return padded, mask

