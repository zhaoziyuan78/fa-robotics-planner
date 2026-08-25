"""Single construction path used by training, evaluation, and tests."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .action_adapter import ActionAdapter
from .action_prior import CausalActionPrior
from .encoders import VisualEncoder
from .method import FunctionAlignmentWM
from .state_adapter import StateAdapter
from .state_prior import CausalStatePrior


def build_method(config: Mapping[str, Any], action_low=None, action_high=None) -> FunctionAlignmentWM:
    env = config["env"]
    model = config["model"]
    action_dim = int(env["action_size"])
    state_dim = int(env["state_size"])
    goal_dim = int(env.get("goal_size", 0))
    proprio_dim = int(env.get("proprio_size", 0))
    action_cfg = model["action_prior"]
    state_cfg = model["state_prior"]
    low = np.full(action_dim, -1.0, np.float32) if action_low is None else action_low
    high = np.full(action_dim, 1.0, np.float32) if action_high is None else action_high
    action_prior = CausalActionPrior(
        action_dim,
        d_model=int(action_cfg["d_model"]),
        n_layers=int(action_cfg["n_layers"]),
        n_heads=int(action_cfg["n_heads"]),
        dropout=float(action_cfg.get("dropout", 0.1)),
        max_length=int(action_cfg.get("max_length", 128)),
        low=np.asarray(low).tolist(),
        high=np.asarray(high).tolist(),
    )
    state_prior = CausalStatePrior(
        state_dim,
        proprio_dim=proprio_dim,
        visual_dim=int(state_cfg.get("visual_dim", 0)),
        d_model=int(state_cfg["d_model"]),
        n_layers=int(state_cfg["n_layers"]),
        n_heads=int(state_cfg["n_heads"]),
        dropout=float(state_cfg.get("dropout", 0.1)),
        max_length=int(state_cfg.get("max_length", 128)),
        normalize_visual=bool(state_cfg.get("normalize_visual", False)),
        residual_prediction=bool(state_cfg.get("residual_prediction", False)),
    )
    state_adapter_cfg = model["state_adapter"]
    action_adapter_cfg = model["action_adapter"]
    state_adapter = StateAdapter(
        state_dim,
        action_dim,
        int(state_adapter_cfg["hidden_dim"]),
        int(state_cfg["d_model"]),
        residual=not bool(state_adapter_cfg.get("legacy_direct_next_state", False)),
    )
    achieved_bounds = env.get("action_adapter_achieved_goal_slice")
    action_adapter = ActionAdapter(
        action_dim,
        state_dim,
        goal_dim,
        int(action_cfg["d_model"]),
        int(action_adapter_cfg["hidden_dim"]),
        achieved_goal_slice=(int(achieved_bounds[0]), int(achieved_bounds[1]))
        if achieved_bounds is not None
        else None,
        normalize_goal_direction=bool(
            action_adapter_cfg.get("normalize_goal_direction", False)
        ),
        residual_scale=float(action_adapter_cfg.get("residual_scale", 0.5)),
        residual_clip=(
            None
            if action_adapter_cfg.get("residual_clip", 4.0) is None
            else float(action_adapter_cfg.get("residual_clip", 4.0))
        ),
        goal_feature_scale=float(action_adapter_cfg.get("goal_feature_scale", 1.0)),
    )
    method = FunctionAlignmentWM(
        action_prior,
        state_prior,
        state_adapter,
        action_adapter,
        use_state_adapter=bool(state_adapter_cfg.get("enabled", True)),
        use_action_adapter=bool(action_adapter_cfg.get("enabled", True)),
    )
    visual_dim = int(state_cfg.get("visual_dim", 0))
    method.visual_encoder = (
        VisualEncoder(visual_dim, normalize_output=bool(state_cfg.get("normalize_visual", False)))
        if visual_dim
        else None
    )
    if method.visual_encoder is not None:
        method.visual_encoder.eval()
        for parameter in method.visual_encoder.parameters():
            parameter.requires_grad = False
    return method
