"""Gymnasium-Robotics Fetch wrappers with version fallback and OOD controls."""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from .unified import ObservationBundle, StepResult, UnifiedControlEnv, pad_vector


FETCH_IDS = {
    "fetch_slide": ("FetchSlide-v4", "FetchSlideDense-v4", "FetchSlide-v3", "FetchSlideDense-v3"),
    "fetch_push": ("FetchPush-v4", "FetchPushDense-v4", "FetchPush-v3", "FetchPushDense-v3"),
}


def _load_gym():
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except ImportError as exc:
        raise RuntimeError(
            "Fetch tasks require `pip install -e '.[fetch]'` in the planner environment"
        ) from exc
    # Gymnasium-Robotics >=1.4 passes output dimensions to MujocoRenderer,
    # while HumanoidBench pins Gymnasium 0.29 whose renderer reads them from
    # model.vis.global_.  Keep both suites in one environment with this narrow
    # constructor compatibility shim rather than forking either simulator.
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
    import inspect

    if "width" not in inspect.signature(MujocoRenderer.__init__).parameters and not getattr(
        MujocoRenderer, "_fa_dimension_compat", False
    ):
        original_init = MujocoRenderer.__init__

        def compatible_init(self, model, data, default_cam_config=None, width=None, height=None):
            if width is not None:
                model.vis.global_.offwidth = int(width)
            if height is not None:
                model.vis.global_.offheight = int(height)
            original_init(self, model, data, default_cam_config)

        MujocoRenderer.__init__ = compatible_init
        MujocoRenderer._fa_dimension_compat = True
    gym.register_envs(gymnasium_robotics)
    return gym


class FetchControlEnv(UnifiedControlEnv):
    def __init__(self, task: str, config: Mapping[str, Any]):
        if task not in FETCH_IDS:
            raise ValueError(f"Unknown Fetch task: {task}")
        self.task = task
        self.config = dict(config)
        self.image_size = tuple(config.get("image_size", [96, 96]))
        self.state_size = int(config.get("state_size", 64))
        os.environ.setdefault("MUJOCO_GL", str(config.get("mujoco_gl", "egl")))
        gym = _load_gym()
        requested = config.get("env_id")
        candidates = (requested,) if requested else FETCH_IDS[task]
        self.env = None
        errors: list[str] = []
        for env_id in candidates:
            try:
                self.env = gym.make(
                    env_id,
                    render_mode="rgb_array",
                    max_episode_steps=int(config.get("episode_horizon", 50)),
                )
                self.env_id = env_id
                break
            except Exception as exc:  # registry/version fallback
                errors.append(f"{env_id}: {exc}")
        if self.env is None:
            raise RuntimeError("No supported Fetch environment could be created:\n" + "\n".join(errors))
        self._raw_observation: Mapping[str, np.ndarray] = {}
        self._ood: dict[str, Any] = {}
        self._physics_reference: dict[str, np.ndarray] | None = None

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self.env.action_space.low, np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self.env.action_space.high, np.float32)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _cache_physics(self) -> None:
        model = self.unwrapped.model
        if self._physics_reference is None:
            self._physics_reference = {
                "body_mass": model.body_mass.copy(),
                "geom_friction": model.geom_friction.copy(),
                "geom_size": model.geom_size.copy(),
                "actuator_gainprm": model.actuator_gainprm.copy(),
            }

    def _object_body_ids(self) -> list[int]:
        model = self.unwrapped.model
        names = []
        for index in range(model.nbody):
            try:
                name = model.body(index).name or ""
            except Exception:
                name = ""
            if "object" in name or "block" in name or "puck" in name:
                names.append(index)
        return names

    def _object_geom_ids(self) -> list[int]:
        model = self.unwrapped.model
        body_ids = set(self._object_body_ids())
        return [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) in body_ids]

    def _apply_ood_physics(self) -> None:
        self._cache_physics()
        assert self._physics_reference is not None
        model = self.unwrapped.model
        for key, reference in self._physics_reference.items():
            getattr(model, key)[...] = reference
        mass = float(self._ood.get("mass_multiplier", self._ood.get("puck_mass_multiplier", 1.0)))
        friction = float(self._ood.get("friction_multiplier", 1.0))
        size = float(self._ood.get("block_size_multiplier", 1.0))
        effectiveness = float(self._ood.get("actuator_effectiveness", 1.0))
        for body_id in self._object_body_ids():
            model.body_mass[body_id] *= mass
        for geom_id in self._object_geom_ids():
            model.geom_friction[geom_id] *= friction
            model.geom_size[geom_id] *= size
        model.actuator_gainprm[...] *= effectiveness

    def _apply_initial_velocity(self, seed: int) -> None:
        speed = float(self._ood.get("initial_object_speed", 0.0))
        if speed <= 0:
            return
        model, data = self.unwrapped.model, self.unwrapped.data
        rng = np.random.default_rng(seed + 9173)
        for joint_id in range(model.njnt):
            try:
                name = model.joint(joint_id).name or ""
            except Exception:
                name = ""
            if "object" in name:
                address = int(model.jnt_dofadr[joint_id])
                dofs = min(2, data.qvel.size - address)
                direction = rng.normal(size=dofs)
                direction /= max(np.linalg.norm(direction), 1e-6)
                data.qvel[address : address + dofs] = direction * speed
                break

    def _bundle(self) -> ObservationBundle:
        raw = self._raw_observation
        observation = np.asarray(raw.get("observation", np.empty(0)), np.float32).reshape(-1)
        achieved = np.asarray(raw.get("achieved_goal", np.empty(0)), np.float32).reshape(-1)
        control, mask = pad_vector(np.concatenate((observation, achieved)), self.state_size)
        return ObservationBundle(
            rgb=self.render(),
            proprio=observation,
            control_state=control,
            state_mask=mask,
            goal=self.get_goal(),
        )

    def reset(self, seed: int = 0) -> ObservationBundle:
        raw, _ = self.env.reset(seed=int(seed))
        self._raw_observation = raw
        self._apply_ood_physics()
        self._apply_initial_velocity(int(seed))
        if hasattr(self.unwrapped, "_get_obs"):
            self._raw_observation = self.unwrapped._get_obs()
        if "goal" in self._ood:
            goal = np.asarray(self._ood["goal"], np.float32)
            if hasattr(self.unwrapped, "goal"):
                self.unwrapped.goal = goal.copy()
            self._raw_observation = dict(raw, desired_goal=goal)
        return self._bundle()

    def prepare_passive_episode(self, rng: np.random.Generator) -> None:
        passive = self.config.get("passive", {})
        mass = passive.get("mass_multiplier", [1.0, 1.0])
        friction = passive.get("friction_multiplier", [1.0, 1.0])
        self._ood = {
            "mass_multiplier": float(rng.uniform(*mass)),
            "friction_multiplier": float(rng.uniform(*friction)),
            # Initial object velocity is an unlabelled external impulse; task
            # action remains exactly zero in every recorded transition.
            "initial_object_speed": float(rng.uniform(0.01, 0.1)),
        }

    def step(self, action: np.ndarray) -> StepResult:
        clipped = np.clip(np.asarray(action, np.float32), self.action_low, self.action_high)
        raw, reward, terminated, truncated, info = self.env.step(clipped)
        self._raw_observation = raw
        success = bool(np.asarray(info.get("is_success", False)).item())
        return StepResult(
            self._bundle(), float(reward), bool(terminated), bool(truncated), {**info, "success": success}
        )

    def render(self) -> np.ndarray:
        frame = np.asarray(self.env.render(), np.uint8)
        if frame.shape[:2] != self.image_size:
            from PIL import Image

            frame = np.asarray(Image.fromarray(frame).resize(self.image_size[::-1]), np.uint8)
        return frame[..., :3]

    def render_goal(self, goal: np.ndarray | None = None) -> np.ndarray:
        """Render a physically consistent goal image without changing env state.

        Goal-image world models need the manipulated object at the desired
        position. Reusing the current RGB frame makes the visual and
        proprioceptive targets contradict each other. MuJoCo state is restored
        exactly after rendering, so this helper is observation-only.
        """

        target = self.get_goal() if goal is None else np.asarray(goal, np.float32)
        if target.size < 3:
            return self.render()
        unwrapped = self.unwrapped
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        try:
            object_qpos = unwrapped._utils.get_joint_qpos(
                unwrapped.model, unwrapped.data, "object0:joint"
            ).copy()
            object_qpos[:3] = target[:3]
            unwrapped._utils.set_joint_qpos(
                unwrapped.model, unwrapped.data, "object0:joint", object_qpos
            )
            unwrapped._utils.mujoco.mj_forward(unwrapped.model, unwrapped.data)
            return self.render().copy()
        finally:
            unwrapped.data.qpos[:] = qpos
            unwrapped.data.qvel[:] = qvel
            unwrapped._utils.mujoco.mj_forward(unwrapped.model, unwrapped.data)

    def get_goal(self) -> np.ndarray:
        return np.asarray(self._raw_observation.get("desired_goal", np.empty(0)), np.float32).copy()

    def get_achieved_goal(self) -> np.ndarray:
        return np.asarray(self._raw_observation.get("achieved_goal", np.empty(0)), np.float32).copy()

    def get_control_state(self) -> np.ndarray:
        return self._bundle().control_state

    def get_state_mask(self) -> np.ndarray:
        return self._bundle().state_mask

    def compute_score(self, predicted_state: np.ndarray, goal: np.ndarray, action_sequence=None) -> float:
        state = np.asarray(predicted_state, np.float32).reshape(-1)
        goal = np.asarray(goal, np.float32).reshape(-1)
        # Unified Fetch state appends achieved_goal after the raw observation.
        achieved_bounds = self.config.get("achieved_goal_slice")
        if achieved_bounds is None:
            achieved = state[-goal.size :] if goal.size else np.empty(0)
        else:
            achieved = state[int(achieved_bounds[0]) : int(achieved_bounds[1])]
        goal_dimensions = int(
            self.config.get("goal_score_dimensions", goal.size)
        )
        goal_error = float(
            np.linalg.norm(
                achieved[:goal_dimensions] - goal[:goal_dimensions]
            )
        )
        score = -goal_error
        if action_sequence is not None:
            score -= float(self.config.get("control_cost", 0.0)) * float(np.square(action_sequence).sum())
        gripper_bounds = self.config.get("gripper_position_slice", [0, 3])
        gripper = state[int(gripper_bounds[0]) : int(gripper_bounds[1])]
        direction = goal[:2] - achieved[:2]
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        approach_target = achieved.copy()
        approach_target[:2] -= float(self.config.get("approach_offset", 0.0)) * direction
        xy_error = float(np.linalg.norm(gripper[:2] - approach_target[:2]))
        approach_target[2] = achieved[2] + (
            float(self.config.get("approach_safe_height", 0.0))
            if xy_error > float(self.config.get("approach_xy_threshold", 0.04))
            else float(self.config.get("approach_contact_height", 0.008))
        )
        approach_distance = float(np.linalg.norm(gripper - approach_target))
        if (
            action_sequence is not None
            and approach_distance
            < float(self.config.get("contact_alignment_distance", 0.06))
        ):
            actions = np.asarray(action_sequence, np.float32).reshape(-1, self.action_low.size)
            aligned_action = max(
                0.0,
                float(np.mean(actions[:, :2] @ direction)),
            )
            score += float(
                self.config.get("contact_alignment_reward", 0.0)
            ) * aligned_action
        if self.task == "fetch_slide" and state.size >= goal.size + 2:
            velocity_bounds = self.config.get(
                "object_velocity_slice", [-goal.size - 2, -goal.size]
            )
            velocity = state[int(velocity_bounds[0]) : int(velocity_bounds[1])]
            if goal_error < float(
                self.config.get("terminal_velocity_distance", float("inf"))
            ):
                score -= float(self.config.get("terminal_velocity_cost", 0.1)) * float(
                    np.linalg.norm(velocity)
                )
            if float(np.dot(velocity[:2], direction)) <= float(
                self.config.get("approach_release_velocity", 0.02)
            ):
                score -= float(self.config.get("approach_cost", 0.0)) * approach_distance
        elif self.task == "fetch_push" and state.size >= 6:
            score -= float(self.config.get("approach_cost", 0.05)) * approach_distance
        return score

    def set_ood_parameters(self, config: Mapping[str, Any]) -> None:
        self._ood = dict(config)

    def close(self) -> None:
        self.env.close()
