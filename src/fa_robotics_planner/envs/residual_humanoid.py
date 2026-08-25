"""Shared nominal-controller residual action interface for HumanoidBench."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


class FrozenNominalPolicy:
    def __init__(self, policy: Callable[[np.ndarray], np.ndarray], source: str):
        self._policy = policy
        self.source = source

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self._policy(np.asarray(observation, np.float32)), np.float32)

    @classmethod
    def load(cls, checkpoint: str | Path, device: str = "cpu") -> "FrozenNominalPolicy":
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Nominal controller checkpoint not found: {path}")
        if path.suffix == ".zip":
            from stable_baselines3 import PPO

            model = PPO.load(path, device=device)

            def predict(obs: np.ndarray) -> np.ndarray:
                action, _ = model.predict(obs, deterministic=True)
                return action

            return cls(predict, str(path))
        if path.suffix == ".npz":
            data = np.load(path)
            weight, bias = np.asarray(data["weight"], np.float32), np.asarray(data["bias"], np.float32)
            return cls(lambda obs: weight @ obs + bias, str(path))
        import torch

        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

        def act(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                output = module(tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output.squeeze(0).cpu().numpy()

        return cls(act, str(path))


class FrozenReachNominalPolicy:
    """Stable H1 body controller from HumanoidBench's reaching policy.

    The released policy controls the 19 non-hand actuators and stabilizes the
    body while tracking a Cartesian left-hand target.  We keep that target at
    its reset position for the task-independent nominal action.  Task-specific
    targets are used only by the offline paired-data expert.
    """

    def __init__(
        self,
        env: Any,
        checkpoint: str | Path,
        mean_path: str | Path,
        variance_path: str | Path,
        device: str = "cpu",
        target_offset: np.ndarray | None = None,
    ):
        import mujoco
        import torch

        class ReachNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense1 = torch.nn.Linear(55, 256)
                self.dense2 = torch.nn.Linear(256, 256)
                self.dense3 = torch.nn.Linear(256, 19)

            def forward(self, value):
                value = torch.tanh(self.dense1(value))
                value = torch.tanh(self.dense2(value))
                return self.dense3(value)

        checkpoint = Path(checkpoint)
        mean_path = Path(mean_path)
        variance_path = Path(variance_path)
        for path in (checkpoint, mean_path, variance_path):
            if not path.exists():
                raise FileNotFoundError(f"Reach nominal-controller asset not found: {path}")
        self.env = env.unwrapped
        self.device = torch.device(device)
        self.model = ReachNetwork().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        # HumanoidBench stores running-stat snapshots; the first row is the
        # fixed normalization used by its exported TorchPolicy.
        self.mean = np.asarray(np.load(mean_path, mmap_mode="r")[0], np.float32)
        self.variance = np.asarray(
            np.load(variance_path, mmap_mode="r")[0], np.float32
        )
        if self.mean.shape != (55,) or self.variance.shape != (55,):
            raise ValueError("Reach normalization statistics must have shape (55,)")
        self.body_position_indices: list[int] = []
        self.body_velocity_indices: list[int] = []
        position_index = velocity_index = 0
        for joint_id in range(self.env.model.njnt):
            name = mujoco.mj_id2name(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if name.startswith("free_"):
                if name == "free_base":
                    self.body_position_indices.extend(range(position_index, position_index + 7))
                    self.body_velocity_indices.extend(range(velocity_index, velocity_index + 6))
                position_index += 7
                velocity_index += 6
            else:
                if not name.startswith(("lh_", "rh_")) and "wrist" not in name:
                    self.body_position_indices.append(position_index)
                    self.body_velocity_indices.append(velocity_index)
                position_index += 1
                velocity_index += 1
        self.actuator_indices = list(range(15)) + list(range(16, 20))
        self.target = np.zeros(3, np.float32)
        self.target_offset = np.asarray(
            np.zeros(3) if target_offset is None else target_offset, np.float32
        ).reshape(3)
        self.source = str(checkpoint)

    def reset(self) -> None:
        self.target = np.asarray(
            self.env.robot.left_hand_position(), np.float32
        ).copy() + self.target_offset

    def _observation(self, target: np.ndarray) -> np.ndarray:
        position = self.env.data.qpos[self.body_position_indices].copy()
        velocity = self.env.data.qvel[self.body_velocity_indices].copy()
        hand = self.env.robot.left_hand_position().copy()
        target = np.asarray(target, np.float32).reshape(3).copy()
        offset = np.asarray([position[0], position[1], 0.0])
        position[:3] -= offset
        hand -= offset
        target -= offset
        observation = np.concatenate((position[2:], velocity, hand, target)).astype(
            np.float32
        )
        if observation.shape != (55,):
            raise ValueError(f"Reach controller expected 55 inputs, got {observation.shape}")
        return observation

    def action_for_target(self, target: np.ndarray) -> np.ndarray:
        import torch

        observation = self._observation(target)
        normalized = (observation - self.mean) / np.sqrt(self.variance + 1e-8)
        with torch.inference_mode():
            body_action = self.model(
                torch.as_tensor(normalized, device=self.device).unsqueeze(0)
            ).squeeze(0).cpu().numpy()
        body_action = np.clip(body_action, -1.0, 1.0)
        low, high = self.env.action_low, self.env.action_high
        physical_body_action = (body_action + 1.0) * 0.5 * (
            high[self.actuator_indices] - low[self.actuator_indices]
        ) + low[self.actuator_indices]
        physical_action = self.env.data.ctrl.copy()
        physical_action[self.actuator_indices] = physical_body_action
        # Match HumanoidBench's official SingleReachWrapper hand posture.
        if physical_action.size > 20:
            physical_action[15] = 1.57
            physical_action[20] = 1.57
        return np.asarray(
            self.env.task.normalize_action(physical_action), np.float32
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self.action_for_target(self.target)


class ResidualHumanoidAction:
    def __init__(
        self,
        nominal_policy: FrozenNominalPolicy,
        low: np.ndarray,
        high: np.ndarray,
        residual_scale: float = 0.25,
    ):
        self.nominal_policy = nominal_policy
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.residual_scale = float(residual_scale)

    def actual_action(self, observation: np.ndarray, residual_action: np.ndarray) -> np.ndarray:
        nominal = self.nominal_policy(observation)
        residual = np.asarray(residual_action, np.float32)
        if nominal.shape != self.low.shape or residual.shape != self.low.shape:
            raise ValueError(
                f"Nominal/residual/action-space shape mismatch: {nominal.shape}, {residual.shape}, {self.low.shape}"
            )
        return np.clip(nominal + self.residual_scale * residual, self.low, self.high).astype(np.float32)
