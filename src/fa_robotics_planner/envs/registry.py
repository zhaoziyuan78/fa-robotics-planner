"""Configuration-driven environment registry (no name branches outside here)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .unified import UnifiedControlEnv


ENV_NAMES = (
    "windy",
    "fetch_slide",
    "fetch_push",
    "humanoid_stand",
    "humanoid_balance",
    "humanoid_reach",
    "humanoid_push",
)


def make_env(config: Mapping[str, Any]) -> UnifiedControlEnv:
    env_config = dict(config.get("env", config))
    name = str(env_config.get("name", ""))
    if name == "windy":
        from .windy import WindyControlEnv

        return WindyControlEnv(env_config)
    if name in {"fetch_slide", "fetch_push"}:
        from .fetch import FetchControlEnv

        return FetchControlEnv(name, env_config)
    if name.startswith("humanoid_"):
        from .humanoid import HumanoidControlEnv

        nominal_config = config.get("humanoid_nominal_controller", {})
        nominal_controller = config.get("nominal_controller_checkpoint") or nominal_config.get(
            "checkpoint"
        )
        if not nominal_controller and config.get("checkpoint_root"):
            nominal_controller = (
                Path(str(config["checkpoint_root"])).expanduser()
                / "infrastructure"
                / "h1hand_stand_ppo.zip"
            )
        if nominal_controller:
            env_config.setdefault("nominal_controller", nominal_controller)
        for source, destination in (
            ("nominal_controller_type", "nominal_controller_type"),
            ("nominal_controller_mean", "nominal_controller_mean"),
            ("nominal_controller_variance", "nominal_controller_variance"),
        ):
            if config.get(source) is not None:
                env_config.setdefault(destination, config[source])
        for source, destination in (
            ("type", "nominal_controller_type"),
            ("mean", "nominal_controller_mean"),
            ("variance", "nominal_controller_variance"),
        ):
            if nominal_config.get(source) is not None:
                env_config.setdefault(destination, nominal_config[source])
        return HumanoidControlEnv(name, env_config)
    raise ValueError(f"Unknown environment {name!r}; choose one of {ENV_NAMES}")
