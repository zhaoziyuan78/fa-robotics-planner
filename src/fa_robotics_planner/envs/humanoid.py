"""HumanoidBench H1 wrappers sharing a padded observation/action schema."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .residual_humanoid import (
    FrozenNominalPolicy,
    FrozenReachNominalPolicy,
    ResidualHumanoidAction,
)
from .unified import ObservationBundle, StepResult, UnifiedControlEnv, pad_vector


HUMANOID_IDS = {
    "humanoid_stand": "h1hand-stand-v0",
    "humanoid_balance": "h1hand-balance_simple-v0",
    "humanoid_reach": "h1hand-reach-v0",
    "humanoid_push": "h1hand-push-v0",
}


def _flat_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, Mapping):
        parts = [np.asarray(observation[key], np.float32).reshape(-1) for key in sorted(observation)]
        return np.concatenate(parts) if parts else np.empty(0, np.float32)
    return np.asarray(observation, np.float32).reshape(-1)


class HumanoidControlEnv(UnifiedControlEnv):
    def __init__(self, task: str, config: Mapping[str, Any]):
        if task not in HUMANOID_IDS:
            raise ValueError(f"Unknown Humanoid task: {task}")
        try:
            import gymnasium as gym
            import humanoid_bench  # noqa: F401 - registers environments
        except ImportError as exc:
            raise RuntimeError(
                "Humanoid tasks require the HumanoidBench source package in the planner environment"
            ) from exc
        self.task = task
        self.config = dict(config)
        self.image_size = tuple(config.get("image_size", [96, 96]))
        self.state_size = int(config.get("state_size", 512))
        self.nominal_observation_size = int(config.get("nominal_observation_size", 151))
        cache = Path(config.get("matplotlib_cache", "/tmp/fa-robotics-planner-matplotlib"))
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        os.environ.setdefault("MUJOCO_GL", str(config.get("mujoco_gl", "egl")))
        self.env_id = str(config.get("env_id", HUMANOID_IDS[task]))
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        checkpoint = config.get("nominal_controller")
        if not checkpoint:
            raise RuntimeError(
                "Humanoid tasks require env.nominal_controller; zero control is deliberately not used as a silent fallback"
            )
        nominal_type = str(config.get("nominal_controller_type", "ppo"))
        if nominal_type == "reach":
            nominal = FrozenReachNominalPolicy(
                self.env,
                checkpoint,
                config.get("nominal_controller_mean", ""),
                config.get("nominal_controller_variance", ""),
                str(config.get("device", "cpu")),
                np.asarray(config.get("nominal_target_offset", [0.0, 0.0, 0.0])),
            )
        elif nominal_type == "ppo":
            nominal = FrozenNominalPolicy.load(
                checkpoint, str(config.get("device", "cpu"))
            )
        else:
            raise ValueError("env.nominal_controller_type must be 'ppo' or 'reach'")
        self.residual = ResidualHumanoidAction(
            nominal,
            self.env.action_space.low,
            self.env.action_space.high,
            float(config.get("residual_scale", 0.25)),
        )
        self._raw: Any = np.empty(0, np.float32)
        self._flat = np.empty(0, np.float32)
        self._ood: dict[str, Any] = {}
        self._physics_reference: dict[str, np.ndarray] | None = None

    @property
    def action_low(self) -> np.ndarray:
        return np.full_like(self.env.action_space.low, -1.0, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.full_like(self.env.action_space.high, 1.0, dtype=np.float32)

    def _indices(self, key: str) -> np.ndarray:
        indices = self.config.get("state_indices", {}).get(key, [])
        return np.asarray(indices, dtype=np.int64)

    def _field(self, key: str) -> np.ndarray:
        indices = self._indices(key)
        return self._flat[indices] if indices.size else np.empty(0, np.float32)

    def _bundle(self) -> ObservationBundle:
        control, mask = pad_vector(self._flat, self.state_size)
        proprio_indices = self._indices("proprio")
        proprio = (
            self._flat[proprio_indices]
            if proprio_indices.size
            else self._flat[: self.nominal_observation_size]
        )
        return ObservationBundle(self.render(), proprio, control, mask, self.get_goal())

    def _apply_ood(self) -> None:
        unwrapped = self.env.unwrapped
        model = getattr(unwrapped, "model", None)
        if model is None:
            return
        if self._physics_reference is None:
            self._physics_reference = {
                "geom_friction": model.geom_friction.copy(),
                "dof_damping": model.dof_damping.copy(),
                "actuator_gainprm": model.actuator_gainprm.copy(),
            }
        for key, value in self._physics_reference.items():
            getattr(model, key)[...] = value
        model.geom_friction[...] *= float(self._ood.get("friction_multiplier", 1.0))
        model.dof_damping[...] *= float(self._ood.get("damping_multiplier", 1.0))
        model.actuator_gainprm[...] *= float(self._ood.get("actuator_strength_multiplier", 1.0))
        impulse = self._ood.get("torso_impulse")
        data = getattr(unwrapped, "data", None)
        if impulse is not None and data is not None and hasattr(data, "qvel"):
            impulse = np.asarray(impulse, np.float32).reshape(-1)
            data.qvel[: min(impulse.size, 3)] += impulse[:3]

    def reset(self, seed: int = 0) -> ObservationBundle:
        # HumanoidBench Reach/Push currently sample goals through the global
        # NumPy RNG rather than env.np_random.
        np.random.seed(int(seed))
        self.env.action_space.seed(int(seed))
        self._raw, _ = self.env.reset(seed=int(seed))
        self._flat = _flat_observation(self._raw)
        reset_nominal = getattr(self.residual.nominal_policy, "reset", None)
        if callable(reset_nominal):
            reset_nominal()
        self._expert_target = np.asarray(
            getattr(
                self.residual.nominal_policy,
                "target",
                self.env.unwrapped.robot.left_hand_position(),
            ),
            np.float32,
        ).copy()
        self._apply_ood()
        task = getattr(self.env.unwrapped, "task", None)
        if task is not None and hasattr(task, "get_obs"):
            self._raw = task.get_obs()
            self._flat = _flat_observation(self._raw)
        return self._bundle()

    def prepare_passive_episode(self, rng: np.random.Generator) -> None:
        passive = self.config.get("passive", {})

        def sample(name: str, default: tuple[float, float]) -> float:
            bounds = passive.get(name, default)
            return float(rng.uniform(float(bounds[0]), float(bounds[1])))

        impulse = rng.normal(0.0, float(passive.get("impulse_std", 0.05)), size=3)
        self._ood = {
            "friction_multiplier": sample("friction_multiplier", (0.8, 1.2)),
            "damping_multiplier": sample("damping_multiplier", (0.9, 1.1)),
            "actuator_strength_multiplier": sample("actuator_strength_multiplier", (0.9, 1.1)),
            "torso_impulse": impulse.astype(np.float32),
        }

    def step(self, action: np.ndarray) -> StepResult:
        actual = self.residual.actual_action(
            self._flat[: self.nominal_observation_size], np.clip(action, -1.0, 1.0)
        )
        self._raw, reward, terminated, truncated, info = self.env.step(actual)
        self._flat = _flat_observation(self._raw)
        success = any(
            bool(info.get(key, False))
            for key in self.config.get(
                "success_info_keys", ["success", "is_success"]
            )
        )
        return StepResult(self._bundle(), float(reward), bool(terminated), bool(truncated), {**info, "success": success, "actual_action": actual})

    def episode_success(
        self,
        success: bool,
        episode_return: float,
        steps_taken: int,
        episode_horizon: int,
        terminated: bool,
        truncated: bool,
    ) -> bool:
        del truncated
        mode = str(self.config.get("episode_success_mode", "info"))
        if mode == "info":
            return bool(success)
        if mode == "survival":
            return bool(success) or (
                not terminated and int(steps_taken) >= int(episode_horizon)
            )
        if mode == "return":
            threshold = float(self.config["success_return_threshold"])
            native_horizon = int(
                self.config.get("success_return_native_horizon", episode_horizon)
            )
            scaled_threshold = threshold * int(episode_horizon) / native_horizon
            return bool(success) or float(episode_return) >= scaled_threshold
        raise ValueError(
            "env.episode_success_mode must be 'info', 'survival', or 'return'"
        )

    def render(self) -> np.ndarray:
        frame = np.asarray(self.env.render(), np.uint8)[..., :3]
        if frame.shape[:2] != self.image_size:
            from PIL import Image

            frame = np.asarray(Image.fromarray(frame).resize(self.image_size[::-1]), np.uint8)
        return frame

    def get_goal(self) -> np.ndarray:
        if isinstance(self._raw, Mapping) and "desired_goal" in self._raw:
            return np.asarray(self._raw["desired_goal"], np.float32).reshape(-1)
        if self.task == "humanoid_reach":
            return self._flat[154:157].copy()
        if self.task == "humanoid_push":
            return self._flat[154:157].copy()
        return self._field("goal").copy()

    def get_achieved_goal(self) -> np.ndarray:
        if isinstance(self._raw, Mapping) and "achieved_goal" in self._raw:
            return np.asarray(self._raw["achieved_goal"], np.float32).reshape(-1)
        if self.task == "humanoid_reach":
            return self._flat[151:154].copy()
        if self.task == "humanoid_push":
            return self._flat[157:160].copy()
        return self._field("achieved_goal").copy()

    def get_control_state(self) -> np.ndarray:
        return self._bundle().control_state

    def paired_expert_action(self, observation: ObservationBundle) -> np.ndarray:
        """Task controller used only to label offline Humanoid paired data."""

        nominal = self.residual.nominal_policy
        action_for_target = getattr(nominal, "action_for_target", None)
        if not callable(action_for_target) or self.task not in {
            "humanoid_reach",
            "humanoid_push",
        }:
            return np.zeros_like(self.action_low)
        nominal_action = nominal(self._flat[: self.nominal_observation_size])
        if self.task == "humanoid_reach":
            hand = self._flat[151:154]
            direction = observation.goal - hand
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            target = observation.goal + float(
                self.config.get("expert_target_overshoot", 0.05)
            ) * direction
        else:
            obj = self.get_achieved_goal()
            direction = observation.goal - obj
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            hand = self._flat[151:154]
            behind = obj - 0.12 * direction
            behind[2] = obj[2] + 0.08
            target = (
                behind
                if np.linalg.norm(hand - behind) > 0.12
                else obj + 0.18 * direction
            )
        target_step = float(self.config.get("expert_target_step", 0.1))
        self._expert_target += np.clip(
            np.asarray(target, np.float32) - self._expert_target,
            -target_step,
            target_step,
        )
        expert_action = action_for_target(self._expert_target)
        return np.clip(
            (expert_action - nominal_action) / max(self.residual.residual_scale, 1e-6),
            self.action_low,
            self.action_high,
        ).astype(np.float32)

    def get_state_mask(self) -> np.ndarray:
        return self._bundle().state_mask

    def compute_score(self, predicted_state: np.ndarray, goal: np.ndarray, action_sequence=None) -> float:
        from fa_robotics_planner.planning.scorers import humanoid_score

        return humanoid_score(self.task, predicted_state, goal, action_sequence, self.config.get("scorer", {}))

    def set_ood_parameters(self, config: Mapping[str, Any]) -> None:
        self._ood = dict(config)

    def close(self) -> None:
        self.env.close()


def assert_shared_humanoid_action_space(envs: Sequence[HumanoidControlEnv]) -> None:
    if not envs:
        raise ValueError("At least one humanoid environment is required")
    shape = envs[0].action_low.shape
    low, high = envs[0].action_low, envs[0].action_high
    for env in envs[1:]:
        if env.action_low.shape != shape:
            raise AssertionError(f"Humanoid action shape differs: {env.env_id}")
        if not np.allclose(env.action_low, low) or not np.allclose(env.action_high, high):
            raise AssertionError(f"Humanoid normalized action bounds differ: {env.env_id}")
