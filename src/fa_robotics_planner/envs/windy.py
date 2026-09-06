"""WindyNav source-of-truth dynamics behind the unified API."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .rendering import render_windy
from .unified import ObservationBundle, StepResult, UnifiedControlEnv


DEFAULT_COLORS = [[60, 70, 80], [70, 80, 70], [80, 70, 60], [70, 60, 80]]


class WindyControlEnv(UnifiedControlEnv):
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.dt = float(config.get("dt", 0.1))
        self.horizon = int(config.get("episode_horizon", config.get("T", 100)))
        self.gamma = float(config.get("gamma", 0.9))
        self.v_max = float(config.get("v_max", 1.0))
        self.a_max = float(config.get("a_max", 1.0))
        self.w_max = float(config.get("w_max", 0.6))
        self.bounce_beta = float(config.get("bounce_beta", 0.5))
        self.success_radius = float(config.get("success_radius", 0.1))
        self.image_size = tuple(config.get("image_size", [64, 64]))
        self.region_count = int(config.get("region_count", 4))
        self.region_w_scale = float(config.get("region_w_scale", 0.6))
        self.region_colors = config.get("region_colors", DEFAULT_COLORS)
        self.edge_margin = float(config.get("edge_margin", 0.08))
        self.legacy_static_wind = bool(config.get("legacy_static_wind", True))
        self._ood: dict[str, Any] = {}
        self._rng = np.random.default_rng(0)
        self.state = np.zeros(4, np.float32)
        self.goal = np.zeros(2, np.float32)
        self.region_w0 = np.zeros((self.region_count, 2), np.float32)
        self.region_k = np.zeros((self.region_count, 2), np.float32)
        self.t = 0

    @property
    def action_low(self) -> np.ndarray:
        return np.full(2, -self.a_max, np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.full(2, self.a_max, np.float32)

    def _bundle(self) -> ObservationBundle:
        state = self.state.astype(np.float32, copy=True)
        return ObservationBundle(
            rgb=self.render(),
            proprio=state[2:],
            control_state=state,
            state_mask=np.ones(4, bool),
            goal=self.goal.copy(),
        )

    def reset(self, seed: int = 0, start_mode: str | None = None) -> ObservationBundle:
        self._rng = np.random.default_rng(int(seed))
        start_mode = start_mode or self._ood.get(
            "start_mode", self.config.get("start_mode", "edge")
        )
        if start_mode == "edge":
            position = np.array(
                [self._rng.uniform(-0.8, 0.8), 1.0 - self.edge_margin], np.float32
            )
        elif start_mode == "random":
            position = self._rng.uniform(-0.8, 0.8, 2).astype(np.float32)
        else:
            raise ValueError(f"Unknown start_mode: {start_mode}")
        self.state = np.concatenate((position, np.zeros(2, np.float32)))
        self.goal = np.asarray(
            self._ood.get(
                "goal",
                [self._rng.uniform(-0.8, 0.8), -1.0 + self.edge_margin],
            ),
            dtype=np.float32,
        )
        wind_scale = float(self._ood.get("wind_scale", 1.0))
        self.region_w0 = self._rng.uniform(
            -self.w_max * self.region_w_scale * wind_scale,
            self.w_max * self.region_w_scale * wind_scale,
            (self.region_count, 2),
        ).astype(np.float32)
        slope = float(self._ood.get("slope", 0.0))
        self.region_k = np.full((self.region_count, 2), slope, np.float32)
        quadrant = self._ood.get("wind_quadrant")
        if quadrant is not None:
            signs = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}[int(quadrant)]
            self.region_w0 = np.abs(self.region_w0) * np.asarray(signs, np.float32)
        self.t = 0
        return self._bundle()

    def _region_index(self, y: float) -> int:
        return min(max(int(((y + 1.0) / 2.0) * self.region_count), 0), self.region_count - 1)

    def _wind_at(self, y: float) -> np.ndarray:
        index = self._region_index(y)
        if self.legacy_static_wind:
            wind = self.region_w0[index]
        else:
            sign = -1.0 if self._ood.get("reverse_slope_at") is not None and self.t >= int(self._ood["reverse_slope_at"]) else 1.0
            wind = self.region_w0[index] + sign * self.region_k[index] * self.t
        cap = min(self.w_max * float(self._ood.get("wind_cap_multiplier", 1.0)), 2 * self.w_max)
        return np.clip(wind, -cap, cap).astype(np.float32)

    def current_wind(self) -> np.ndarray:
        return self._wind_at(float(self.state[1]))

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, np.float32), self.action_low, self.action_high)
        position, velocity = self.state[:2].copy(), self.state[2:].copy()
        velocity = np.clip(self.gamma * velocity + action * self.dt, -self.v_max, self.v_max)
        wind = self._wind_at(float(position[1]))
        position = position + (velocity + wind) * self.dt
        for dimension in range(2):
            if abs(position[dimension]) > 1.0:
                position[dimension] = np.sign(position[dimension])
                velocity[dimension] = -self.bounce_beta * velocity[dimension]
        self.state = np.concatenate((position, velocity)).astype(np.float32)
        self.t += 1
        distance = float(np.linalg.norm(position - self.goal))
        success = distance <= self.success_radius
        terminate_on_success = bool(
            getattr(self, "terminate_on_success", True)
        )
        return StepResult(
            self._bundle(),
            reward=float(success),
            terminated=bool(success and terminate_on_success),
            truncated=self.t >= self.horizon and not (success and terminate_on_success),
            info={"success": success, "distance": distance, "wind": wind.copy()},
        )

    def render(self) -> np.ndarray:
        if self.image_size[0] != self.image_size[1]:
            raise ValueError("Windy renderer requires a square image")
        return render_windy(
            self.image_size[0], self.state[:2], self.state[2:], self.goal, self.region_colors
        )

    def get_goal(self) -> np.ndarray:
        return self.goal.copy()

    def get_achieved_goal(self) -> np.ndarray:
        return self.state[:2].copy()

    def get_control_state(self) -> np.ndarray:
        return self.state.copy()

    def compute_score(self, predicted_state: np.ndarray, goal: np.ndarray, action_sequence=None) -> float:
        state = np.asarray(predicted_state)
        return -float(np.linalg.norm(state[..., :2] - np.asarray(goal)))

    def set_ood_parameters(self, config: Mapping[str, Any]) -> None:
        self._ood = dict(config)
