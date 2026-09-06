"""Single construction path used by training, evaluation, and tests."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .action_adapter import ActionAdapter
from .action_prior import CausalActionPrior
from .method import FunctionAlignmentWM
from .state_adapter import StateAdapter
from .state_prior import CausalStatePrior
from .vqvae import VQVAE


def build_method(config: Mapping[str, Any], action_low=None, action_high=None) -> FunctionAlignmentWM:
    env = config["env"]
    model = config["model"]
    action_dim = int(env["action_size"])
    state_dim = int(env["state_size"])
    goal_dim = int(env.get("goal_size", 0))
    action_cfg = model["action_prior"]
    state_cfg = model["state_prior"]
    tokenizer_cfg = model["tokenizer"]
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
    image_height, image_width = map(int, env.get("image_size", (64, 64)))
    downsample = int(tokenizer_cfg.get("downsample_factor", 8))
    if image_height % downsample or image_width % downsample:
        raise ValueError("Environment image_size must be divisible by tokenizer downsample_factor")
    tokens_per_frame = (image_height // downsample) * (image_width // downsample)
    video_cfg = state_cfg["video"]
    observation_cfg = state_cfg["observation"]
    state_prior = CausalStatePrior(
        state_dim,
        codebook_size=int(tokenizer_cfg["codebook_size"]),
        tokens_per_frame=tokens_per_frame,
        video_d_model=int(video_cfg["d_model"]),
        video_layers=int(video_cfg["n_layers"]),
        video_heads=int(video_cfg["n_heads"]),
        video_d_ff=int(video_cfg.get("d_ff", 4 * int(video_cfg["d_model"]))),
        observation_type=str(observation_cfg.get("type", "auto")),
        observation_d_model=int(observation_cfg["d_model"]),
        observation_layers=int(observation_cfg["n_layers"]),
        observation_heads=int(observation_cfg.get("n_heads", 4)),
        dropout=float(state_cfg.get("dropout", 0.1)),
        context_frames=int(state_cfg.get("context_frames", 8)),
        max_frames=int(state_cfg.get("max_frames", 128)),
    )
    state_adapter_cfg = model["state_adapter"]
    action_adapter_cfg = model["action_adapter"]
    state_adapter = StateAdapter(
        state_dim,
        action_dim,
        int(state_adapter_cfg["hidden_dim"]),
        state_prior.observation_hidden_dim,
        state_prior.video_hidden_dim,
        state_prior.codebook_size,
        state_prior.tokens_per_frame,
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
        VQVAE(
            in_channels=3,
            hidden_dim=int(tokenizer_cfg.get("hidden_dim", 128)),
            codebook_size=int(tokenizer_cfg["codebook_size"]),
            code_dim=int(tokenizer_cfg.get("code_dim", 128)),
            commitment_weight=float(tokenizer_cfg.get("commitment_weight", 0.25)),
        ),
        use_state_adapter=bool(state_adapter_cfg.get("enabled", True)),
        use_action_adapter=bool(action_adapter_cfg.get("enabled", True)),
    )
    return method
