"""VQ-token Trajectory Transformer with model-predictive beam search.

The baseline intentionally has no access to ``control_state`` at inference.
Each RGB observation is encoded by the frozen VQ-VAE used to build the token
cache. A causal Transformer predicts a discretised action and, conditioned on
that action, the next image tokens, reward, and termination flag. Beam search
uses those learned transitions and rewards to choose the first action.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fa_robotics_planner.baselines.offline_data import (
    OfflineEpisode,
    OfflinePairedData,
    load_paired_data,
)
from fa_robotics_planner.envs.unified import UnifiedControlEnv
from fa_robotics_planner.models.vqvae import VQVAE


def _returns_to_go(rewards: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


def _normalization(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-4).astype(np.float32)


def _finite_range(values: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Cannot discretise non-finite values")
    if high - low < 1e-6:
        radius = max(0.5, abs(low) * 0.1)
        return low - radius, high + radius
    padding = (high - low) * float(margin)
    return low - padding, high + padding


@dataclass(frozen=True)
class TrajectoryDiscretizer:
    """Uniform bins for actions, native rewards, and return-to-go."""

    action_bins: int
    reward_bins: int
    return_bins: int
    reward_low: float
    reward_high: float
    return_low: float
    return_high: float

    @staticmethod
    def _encode(values: np.ndarray, bins: int, low: float, high: float) -> np.ndarray:
        scaled = (np.asarray(values, np.float32) - low) / max(high - low, 1e-6)
        return np.rint(np.clip(scaled, 0.0, 1.0) * (bins - 1)).astype(np.int64)

    @staticmethod
    def _decode(tokens: np.ndarray | torch.Tensor, bins: int, low: float, high: float):
        return low + tokens * ((high - low) / max(1, bins - 1))

    def encode_actions(self, actions: np.ndarray) -> np.ndarray:
        return self._encode(actions, self.action_bins, -1.0, 1.0)

    def decode_actions(self, tokens: np.ndarray | torch.Tensor):
        return self._decode(tokens, self.action_bins, -1.0, 1.0)

    def encode_rewards(self, rewards: np.ndarray) -> np.ndarray:
        return self._encode(rewards, self.reward_bins, self.reward_low, self.reward_high)

    def decode_rewards(self, tokens: np.ndarray | torch.Tensor):
        return self._decode(tokens, self.reward_bins, self.reward_low, self.reward_high)

    def encode_returns(self, returns: np.ndarray) -> np.ndarray:
        return self._encode(returns, self.return_bins, self.return_low, self.return_high)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "action_bins": self.action_bins,
            "reward_bins": self.reward_bins,
            "return_bins": self.return_bins,
            "reward_low": self.reward_low,
            "reward_high": self.reward_high,
            "return_low": self.return_low,
            "return_high": self.return_high,
        }


@dataclass(frozen=True)
class TokenTrajectoryEpisode:
    state_tokens: np.ndarray
    next_state_tokens: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    goals: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def _truncate_after_first_success(
    data: OfflinePairedData,
    success_reward_threshold: float | None,
) -> tuple[OfflinePairedData, int]:
    """Crop only the private TT view; never mutate shared paired shards."""

    episodes: list[OfflineEpisode] = []
    removed = 0
    for episode in data.episodes:
        stop = episode.length
        if success_reward_threshold is not None:
            successes = np.flatnonzero(
                episode.rewards >= float(success_reward_threshold)
            )
            if successes.size:
                stop = int(successes[0]) + 1
        else:
            terminals = np.flatnonzero(episode.dones > 0.5)
            if terminals.size:
                stop = int(terminals[0]) + 1
        removed += episode.length - stop
        dones = episode.dones[:stop].copy()
        if dones.size:
            dones[-1] = 1.0
        episodes.append(
            OfflineEpisode(
                states=episode.states[:stop],
                actions=episode.actions[:stop],
                next_states=episode.next_states[:stop],
                rewards=episode.rewards[:stop],
                dones=dones,
                goals=episode.goals[:stop],
                rgb=None,
                next_rgb=None,
                action_is_expert=None,
            )
        )
    return (
        OfflinePairedData(
            root=data.root,
            episodes=tuple(episodes),
            requested_transitions=data.requested_transitions,
            manifest_transitions=data.manifest_transitions,
        ),
        removed,
    )


def _resolve_token_data(
    paired_root: Path, config: Mapping[str, Any], seed: int
) -> Path:
    explicit = config.get("tokens") or config.get("token_data")
    root = (
        Path(str(explicit)).expanduser()
        if explicit
        else paired_root.parent / "tokens" / f"paired_seed{int(seed)}"
    ).resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"TT VQ-token cache not found: {root / 'manifest.json'}. Run "
            f"`python -m scripts.tokenize_frames --data {paired_root} --output {root} "
            "--vqvae /path/to/vqvae.pt` first, or set baseline.tokens."
        )
    return root


def _load_token_episodes(
    data: OfflinePairedData, token_root: Path
) -> tuple[tuple[TokenTrajectoryEpisode, ...], dict[str, Any]]:
    paired_manifest = json.loads(
        (data.root / "manifest.json").read_text(encoding="utf-8")
    )
    token_manifest = json.loads(
        (token_root / "manifest.json").read_text(encoding="utf-8")
    )
    if token_manifest.get("kind") != "tokens":
        raise ValueError(f"TT token data must have kind='tokens': {token_root}")
    metadata = dict(token_manifest.get("metadata", {}))
    source = metadata.get("source")
    if source and Path(str(source)).expanduser().resolve() != data.root.resolve():
        raise ValueError(
            "TT token cache was produced from a different paired dataset: "
            f"{source} != {data.root}"
        )
    paired_entries = sorted(
        paired_manifest.get("episodes", []), key=lambda item: int(item["id"])
    )
    token_entries = {
        int(item["id"]): item for item in token_manifest.get("episodes", [])
    }
    episodes: list[TokenTrajectoryEpisode] = []
    token_shape: tuple[int, ...] | None = None
    for paired_entry, episode in zip(paired_entries, data.episodes):
        episode_id = int(paired_entry["id"])
        if episode_id not in token_entries:
            raise ValueError(f"Token cache is missing paired episode {episode_id}")
        entry = token_entries[episode_id]
        with np.load(token_root / entry["file"], allow_pickle=False) as shard:
            if "next_video_tokens" not in shard.files:
                raise ValueError(
                    f"TT requires next_video_tokens, missing from {entry['file']}"
                )
            available = int(np.asarray(shard["sequence_length"]).item())
            if available < episode.length:
                raise ValueError(
                    f"Token episode {episode_id} has {available} steps but paired view "
                    f"requires {episode.length}"
                )
            states = np.asarray(
                shard["video_tokens"][: episode.length], np.int64
            ).copy()
            next_states = np.asarray(
                shard["next_video_tokens"][: episode.length], np.int64
            ).copy()
        if states.ndim != 3 or next_states.shape != states.shape:
            raise ValueError(
                f"Expected [T,H,W] current/next tokens in {entry['file']}, got "
                f"{states.shape} and {next_states.shape}"
            )
        if token_shape is None:
            token_shape = tuple(states.shape[1:])
        elif tuple(states.shape[1:]) != token_shape:
            raise ValueError("Inconsistent VQ token grids across episodes")
        episodes.append(
            TokenTrajectoryEpisode(
                state_tokens=states.reshape(episode.length, -1),
                next_state_tokens=next_states.reshape(episode.length, -1),
                actions=episode.actions,
                rewards=episode.rewards,
                dones=episode.dones,
                goals=episode.goals,
            )
        )
    if len(episodes) != len(data.episodes):
        raise RuntimeError("Paired/token episode alignment ended prematurely")
    if not episodes:
        raise ValueError("TT token dataset is empty")
    codebook_size = int(metadata.get("codebook_size", 0))
    maximum_token = max(
        int(max(ep.state_tokens.max(), ep.next_state_tokens.max())) for ep in episodes
    )
    if codebook_size <= maximum_token:
        raise ValueError(
            f"Token id {maximum_token} exceeds manifest codebook size {codebook_size}"
        )
    return tuple(episodes), {**metadata, "token_shape": token_shape}


def _load_tokenizer_bundle(
    token_metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Path]:
    explicit = config.get("vqvae") or config.get("vqvae_checkpoint")
    checkpoint = explicit or token_metadata.get("vqvae")
    if not checkpoint:
        raise ValueError(
            "Token manifest has no VQ-VAE checkpoint; set baseline.vqvae explicitly"
        )
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TT VQ-VAE checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "vqvae" or "tokenizer" not in payload:
        raise ValueError(f"Not a VQ-VAE checkpoint: {checkpoint_path}")
    tokenizer_config = dict(payload.get("config", {}).get("model", {}).get("tokenizer", {}))
    tokenizer_config = {
        "in_channels": int(tokenizer_config.get("in_channels", 3)),
        "hidden_dim": int(tokenizer_config.get("hidden_dim", 128)),
        "codebook_size": int(tokenizer_config.get("codebook_size", 512)),
        "code_dim": int(tokenizer_config.get("code_dim", 128)),
        "commitment_weight": float(tokenizer_config.get("commitment_weight", 0.25)),
    }
    if tokenizer_config["codebook_size"] != int(token_metadata["codebook_size"]):
        raise ValueError("VQ checkpoint and token-cache codebook sizes do not match")
    state = {name: value.detach().cpu() for name, value in payload["tokenizer"].items()}
    return tokenizer_config, state, checkpoint_path


class TrajectoryTransformer(nn.Module):
    """Causal trajectory model over VQ image states and discrete actions."""

    def __init__(
        self,
        num_state_tokens: int,
        codebook_size: int,
        goal_dim: int,
        action_dim: int,
        action_bins: int,
        reward_bins: int,
        return_bins: int,
        d_model: int,
        state_embed_dim: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        context_length: int,
    ) -> None:
        super().__init__()
        self.num_state_tokens = int(num_state_tokens)
        self.codebook_size = int(codebook_size)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.action_bins = int(action_bins)
        self.reward_bins = int(reward_bins)
        self.return_bins = int(return_bins)
        self.d_model = int(d_model)
        self.context_length = int(context_length)
        self.state_token_embed = nn.Embedding(codebook_size, state_embed_dim)
        self.spatial_embed = nn.Embedding(num_state_tokens, state_embed_dim)
        self.state_project = nn.Linear(num_state_tokens * state_embed_dim, d_model)
        self.goal_embed = nn.Linear(goal_dim, d_model, bias=False) if goal_dim else None
        self.return_embed = nn.Embedding(return_bins, d_model)
        # Each (dimension, value-bin) pair owns an embedding. A shared value
        # embedding plus an averaged dimension embedding would lose which
        # actuator received which value, especially for 61-D Humanoid actions.
        self.action_token_embed = nn.Embedding(
            action_dim * (action_bins + 1), d_model
        )
        self.action_dimension_embed = nn.Embedding(action_dim, d_model)
        self.reward_embed = nn.Embedding(reward_bins + 1, d_model)
        self.done_embed = nn.Embedding(3, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.action_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_bins)
        self.transition_fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.next_state_head = nn.Linear(d_model, num_state_tokens * codebook_size)
        self.reward_head = nn.Linear(d_model, reward_bins)
        self.done_head = nn.Linear(d_model, 2)

    def _state_embedding(self, state_tokens: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_embed.weight.view(
            1, 1, self.num_state_tokens, -1
        )
        embedded = self.state_token_embed(state_tokens.long()) + spatial
        return self.state_project(embedded.flatten(-2))

    def _action_embedding(self, action_tokens: torch.Tensor) -> torch.Tensor:
        dimension_offsets = torch.arange(
            self.action_dim, device=action_tokens.device
        ) * (self.action_bins + 1)
        embedded = self.action_token_embed(
            action_tokens.long() + dimension_offsets
        )
        return embedded.mean(dim=-2)

    def hidden(
        self,
        state_tokens: torch.Tensor,
        goals: torch.Tensor,
        return_tokens: torch.Tensor,
        previous_action_tokens: torch.Tensor,
        previous_reward_tokens: torch.Tensor,
        previous_done_tokens: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = state_tokens.shape[1]
        if steps > self.context_length:
            raise ValueError(f"Sequence {steps} exceeds context {self.context_length}")
        positions = torch.arange(steps, device=state_tokens.device)
        hidden = (
            self._state_embedding(state_tokens)
            + self.return_embed(return_tokens.long())
            + self._action_embedding(previous_action_tokens)
            + self.reward_embed(previous_reward_tokens.long())
            + self.done_embed(previous_done_tokens.long())
            + self.position_embed(positions).unsqueeze(0)
        )
        if self.goal_embed is not None:
            hidden = hidden + self.goal_embed(goals.float())
        causal = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=state_tokens.device),
            diagonal=1,
        )
        padding = ~valid.bool() if valid is not None else None
        return self.transformer(hidden, mask=causal, src_key_padding_mask=padding)

    def action_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        queries = hidden.unsqueeze(-2) + self.action_dimension_embed.weight
        return self.action_head(self.action_norm(queries))

    def transition_logits(
        self, hidden: torch.Tensor, action_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        transition = self.transition_fuse(
            torch.cat((hidden, self._action_embedding(action_tokens)), dim=-1)
        )
        leading = transition.shape[:-1]
        return {
            "next_state": self.next_state_head(transition).view(
                *leading, self.num_state_tokens, self.codebook_size
            ),
            "reward": self.reward_head(transition),
            "done": self.done_head(transition),
        }

    def forward(
        self,
        state_tokens: torch.Tensor,
        goals: torch.Tensor,
        return_tokens: torch.Tensor,
        previous_action_tokens: torch.Tensor,
        previous_reward_tokens: torch.Tensor,
        previous_done_tokens: torch.Tensor,
        current_action_tokens: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.hidden(
            state_tokens,
            goals,
            return_tokens,
            previous_action_tokens,
            previous_reward_tokens,
            previous_done_tokens,
            valid,
        )
        return {
            "action": self.action_logits(hidden),
            **self.transition_logits(hidden, current_action_tokens),
        }


def _sample_batch(
    episodes: Sequence[TokenTrajectoryEpisode],
    returns: Sequence[np.ndarray],
    probability: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
    discretizer: TrajectoryDiscretizer,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
) -> dict[str, np.ndarray]:
    num_tokens = episodes[0].state_tokens.shape[-1]
    action_dim = episodes[0].actions.shape[-1]
    goal_dim = episodes[0].goals.shape[-1]
    batch = {
        "state_tokens": np.zeros((batch_size, context_length, num_tokens), np.int64),
        "next_state_tokens": np.zeros((batch_size, context_length, num_tokens), np.int64),
        "goals": np.zeros((batch_size, context_length, goal_dim), np.float32),
        "action_tokens": np.zeros((batch_size, context_length, action_dim), np.int64),
        "previous_action_tokens": np.full(
            (batch_size, context_length, action_dim), discretizer.action_bins, np.int64
        ),
        "return_tokens": np.zeros((batch_size, context_length), np.int64),
        "reward_tokens": np.zeros((batch_size, context_length), np.int64),
        "previous_reward_tokens": np.full(
            (batch_size, context_length), discretizer.reward_bins, np.int64
        ),
        "dones": np.zeros((batch_size, context_length), np.int64),
        "previous_dones": np.full((batch_size, context_length), 2, np.int64),
        "valid": np.zeros((batch_size, context_length), bool),
    }
    episode_indices = rng.choice(len(episodes), size=batch_size, p=probability)
    for row, episode_index in enumerate(episode_indices):
        episode = episodes[int(episode_index)]
        stop = int(rng.integers(1, episode.length + 1))
        start = max(0, stop - context_length)
        length = stop - start
        destination = slice(0, length)
        batch["state_tokens"][row, destination] = episode.state_tokens[start:stop]
        batch["next_state_tokens"][row, destination] = episode.next_state_tokens[start:stop]
        if goal_dim:
            batch["goals"][row, destination] = (
                episode.goals[start:stop] - goal_mean
            ) / goal_std
        current_actions = discretizer.encode_actions(
            np.clip(episode.actions[start:stop], -1.0, 1.0)
        )
        current_rewards = discretizer.encode_rewards(episode.rewards[start:stop])
        batch["action_tokens"][row, destination] = current_actions
        batch["reward_tokens"][row, destination] = current_rewards
        batch["return_tokens"][row, destination] = discretizer.encode_returns(
            returns[int(episode_index)][start:stop]
        )
        batch["dones"][row, destination] = episode.dones[start:stop].astype(np.int64)
        if start > 0:
            batch["previous_action_tokens"][row, 0] = discretizer.encode_actions(
                np.clip(episode.actions[start - 1], -1.0, 1.0)
            )
            batch["previous_reward_tokens"][row, 0] = discretizer.encode_rewards(
                episode.rewards[start - 1 : start]
            )[0]
            batch["previous_dones"][row, 0] = int(episode.dones[start - 1] > 0.5)
        if length > 1:
            batch["previous_action_tokens"][row, 1:length] = current_actions[:-1]
            batch["previous_reward_tokens"][row, 1:length] = current_rewards[:-1]
            batch["previous_dones"][row, 1:length] = batch["dones"][row, : length - 1]
        batch["valid"][row, destination] = True
    return batch


def _masked_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    extra_dimensions = target.ndim - valid.ndim
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(target.shape)
    if extra_dimensions:
        loss = loss.mean(dim=tuple(range(-extra_dimensions, 0)))
    return (loss * valid.float()).sum() / valid.sum().clamp_min(1)


def train_trajectory_transformer(
    data_path: str | Path,
    config: Mapping[str, Any],
    output: str | Path,
    offline_transitions: int,
    seed: int,
    checkpoint_directory: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    from tqdm.auto import trange

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device(
        str(config.get("device", "cuda")) if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_data = load_paired_data(data_path, int(offline_transitions))
    threshold = config.get("success_reward_threshold")
    data, post_success_transitions_removed = _truncate_after_first_success(
        raw_data, None if threshold is None else float(threshold)
    )
    token_root = _resolve_token_data(data.root, config, seed)
    episodes, token_metadata = _load_token_episodes(data, token_root)
    tokenizer_config, tokenizer_state, tokenizer_path = _load_tokenizer_bundle(
        token_metadata, config
    )
    discount = float(config.get("return_discount", 1.0))
    returns = [_returns_to_go(episode.rewards, discount) for episode in episodes]
    all_returns = np.concatenate(returns)
    episode_returns = np.asarray([value[0] for value in returns], np.float32)
    all_rewards = np.concatenate([episode.rewards for episode in episodes])
    all_goals = np.concatenate([episode.goals for episode in episodes], axis=0)
    reward_low, reward_high = _finite_range(all_rewards)
    return_low, return_high = _finite_range(all_returns)
    discretizer = TrajectoryDiscretizer(
        action_bins=int(config.get("action_bins", 21)),
        reward_bins=int(config.get("reward_bins", 101)),
        return_bins=int(config.get("return_bins", 101)),
        reward_low=reward_low,
        reward_high=reward_high,
        return_low=return_low,
        return_high=return_high,
    )
    if data.goal_dim:
        goal_mean, goal_std = _normalization(all_goals)
    else:
        goal_mean = np.empty(0, np.float32)
        goal_std = np.empty(0, np.float32)
    statistics = {
        "goal_mean": goal_mean,
        "goal_std": goal_std,
        "target_return": np.float32(
            np.quantile(
                episode_returns, float(config.get("target_return_quantile", 0.95))
            )
        ),
    }
    model_config = {
        "num_state_tokens": int(episodes[0].state_tokens.shape[-1]),
        "codebook_size": int(token_metadata["codebook_size"]),
        "goal_dim": data.goal_dim,
        "action_dim": data.action_dim,
        "action_bins": discretizer.action_bins,
        "reward_bins": discretizer.reward_bins,
        "return_bins": discretizer.return_bins,
        "d_model": int(config.get("d_model", 256)),
        "state_embed_dim": int(config.get("state_embed_dim", 64)),
        "n_layers": int(config.get("n_layers", 6)),
        "n_heads": int(config.get("n_heads", 8)),
        "d_ff": int(config.get("d_ff", 1024)),
        "dropout": float(config.get("dropout", 0.1)),
        "context_length": int(config.get("context_length", 20)),
    }
    model = TrajectoryTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    batch_size = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 50))
    explicit_updates = config.get("gradient_updates")
    updates = (
        int(explicit_updates)
        if explicit_updates is not None
        else epochs * max(1, int(np.ceil(data.transition_count / batch_size)))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates), eta_min=float(config.get("min_learning_rate", 1e-5))
    )
    probability = np.asarray([episode.length for episode in episodes], np.float64)
    probability /= probability.sum()
    rng = np.random.default_rng(int(seed))
    weights = {
        "action": float(config.get("action_loss_weight", 2.0)),
        "state": float(config.get("state_loss_weight", 1.0)),
        "reward": float(config.get("reward_loss_weight", 1.0)),
        "done": float(config.get("done_loss_weight", 1.0)),
    }
    history: dict[str, list[float]] = {
        name: [] for name in ("total", "action", "state", "reward", "done")
    }
    amp = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    amp_name = str(config.get("amp_dtype", "bfloat16")).lower()
    if amp_name not in {"bfloat16", "float16"}:
        raise ValueError("baseline.amp_dtype must be bfloat16 or float16")
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    # BF16 has FP32's exponent range and does not need dynamic loss scaling.
    # A scaler remains available for older GPUs explicitly configured for FP16.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and amp_dtype == torch.float16
    )
    progress = trange(updates, desc="offline VQ Trajectory Transformer", leave=False)
    model.train()
    for _ in progress:
        numpy_batch = _sample_batch(
            episodes, returns, probability, batch_size,
            model_config["context_length"], rng, discretizer, goal_mean, goal_std,
        )
        batch = {
            name: torch.as_tensor(value, device=device)
            for name, value in numpy_batch.items()
        }
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp
        ):
            prediction = model(
                batch["state_tokens"], batch["goals"], batch["return_tokens"],
                batch["previous_action_tokens"], batch["previous_reward_tokens"],
                batch["previous_dones"], batch["action_tokens"], batch["valid"],
            )
            action_loss = _masked_cross_entropy(
                prediction["action"], batch["action_tokens"], batch["valid"]
            )
            state_loss = _masked_cross_entropy(
                prediction["next_state"], batch["next_state_tokens"], batch["valid"]
            )
            reward_loss = _masked_cross_entropy(
                prediction["reward"], batch["reward_tokens"], batch["valid"]
            )
            done_loss = _masked_cross_entropy(
                prediction["done"], batch["dones"], batch["valid"]
            )
            total = (
                weights["action"] * action_loss + weights["state"] * state_loss
                + weights["reward"] * reward_loss + weights["done"] * done_loss
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.get("grad_clip", 1.0))
        )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_ran = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
        if optimizer_ran:
            scheduler.step()
        for name, value in zip(
            history, (total, action_loss, state_loss, reward_loss, done_loss)
        ):
            history[name].append(float(value.detach()))
        if len(history["total"]) % max(1, updates // 20) == 0:
            progress.set_postfix(
                total=f"{history['total'][-1]:.3g}",
                action=f"{history['action'][-1]:.3g}",
                state=f"{history['state'][-1]:.3g}",
            )
    planner_config = {
        "horizon": int(config.get("planning_horizon", 5)),
        "beam_width": int(config.get("beam_width", 16)),
        "action_candidates": int(config.get("action_candidates", 16)),
        "action_top_k": int(config.get("action_top_k", 4)),
        "temperature": float(config.get("planning_temperature", 1.0)),
        "action_likelihood_weight": float(config.get("action_likelihood_weight", 0.05)),
        "transition_likelihood_weight": float(config.get("transition_likelihood_weight", 0.01)),
        "reward_weight": float(config.get("planning_reward_weight", 1.0)),
    }
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_target = (
        Path(checkpoint_directory) if checkpoint_directory else target / "checkpoint"
    )
    checkpoint_target.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_target / "final_model.pt"
    torch.save(
        {
            "algorithm": "vq_trajectory_transformer",
            "architecture_version": 2,
            "model": model.state_dict(),
            "model_config": model_config,
            "discretizer": discretizer.to_dict(),
            "statistics": statistics,
            "return_discount": discount,
            "planner_config": planner_config,
            "tokenizer_config": tokenizer_config,
            "tokenizer": tokenizer_state,
            "token_shape": tuple(token_metadata["token_shape"]),
        },
        checkpoint,
    )
    diagnostics = {
        "gradient_updates": updates,
        "training_epochs": epochs if explicit_updates is None else None,
        "loss_first": {
            name: float(np.mean(values[: min(20, len(values))]))
            for name, values in history.items()
        },
        "loss_last": {
            name: float(np.mean(values[-min(20, len(values)) :]))
            for name, values in history.items()
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "offline_transitions": raw_data.transition_count,
        "training_transitions_after_terminal_cropping": data.transition_count,
        "post_success_transitions_removed": post_success_transitions_removed,
        "uses_collector_policy_labels": False,
        "uses_evaluation_transitions_for_training_or_statistics": False,
        "observation": "vq_rgb_tokens+desired_goal+return_to_go+trajectory_history",
        "action_inference": "learned_dynamics_beam_search",
        "transition_prediction": "next_vq_tokens+native_reward+done",
        "trajectory_terminal_rule": "first_success" if threshold is not None else "recorded_done",
        "training_environment_steps": 0,
        "dataset": str(data.root),
        "token_dataset": str(token_root),
        "vqvae_checkpoint": str(tokenizer_path),
        "token_grid": list(token_metadata["token_shape"]),
        "target_return": float(statistics["target_return"]),
        "planner": planner_config,
    }
    (target / "training_history.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return checkpoint, diagnostics


@dataclass
class _Beam:
    states: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[int]
    returns: list[float]
    score: float
    first_action: np.ndarray | None = None


def _top_factorized_actions(
    log_probabilities: torch.Tensor,
    candidates: int,
    top_k_per_dimension: int,
) -> list[tuple[np.ndarray, float]]:
    """Exact k-best products within the retained top bins of each dimension."""

    if log_probabilities.ndim != 2:
        raise ValueError("Action log probabilities must have shape [A,bins]")
    dimensions, bins = log_probabilities.shape
    top_k = min(int(top_k_per_dimension), int(bins))
    values, tokens = torch.topk(log_probabilities, k=top_k, dim=-1)
    values_np = values.detach().float().cpu().numpy()
    tokens_np = tokens.detach().cpu().numpy()
    initial = tuple(0 for _ in range(dimensions))

    def score(ranks: tuple[int, ...]) -> float:
        return float(sum(values_np[index, rank] for index, rank in enumerate(ranks)))

    queue: list[tuple[float, tuple[int, ...]]] = [(-score(initial), initial)]
    visited = {initial}
    result: list[tuple[np.ndarray, float]] = []
    while queue and len(result) < int(candidates):
        negative_score, ranks = heapq.heappop(queue)
        action = np.asarray(
            [tokens_np[index, rank] for index, rank in enumerate(ranks)], np.int64
        )
        result.append((action, -negative_score / max(1, dimensions)))
        for dimension in range(dimensions):
            if ranks[dimension] + 1 >= top_k:
                continue
            neighbour = list(ranks)
            neighbour[dimension] += 1
            neighbour_tuple = tuple(neighbour)
            if neighbour_tuple not in visited:
                visited.add(neighbour_tuple)
                heapq.heappush(queue, (-score(neighbour_tuple), neighbour_tuple))
    return result


class VQBeamPlanner:
    def __init__(
        self,
        model: TrajectoryTransformer,
        discretizer: TrajectoryDiscretizer,
        planner_config: Mapping[str, Any],
        goal_mean: np.ndarray,
        goal_std: np.ndarray,
        return_discount: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.discretizer = discretizer
        self.config = planner_config
        self.goal_mean = goal_mean
        self.goal_std = goal_std
        self.return_discount = float(return_discount)
        self.device = device

    def _last_hidden(self, beams: Sequence[_Beam], goal: np.ndarray) -> torch.Tensor:
        context = self.model.context_length
        state_rows = []
        return_rows = []
        previous_actions = []
        previous_rewards = []
        previous_dones = []
        for beam in beams:
            start = max(0, len(beam.states) - context)
            indices = list(range(start, len(beam.states)))
            state_rows.append(np.stack([beam.states[index] for index in indices]))
            return_rows.append(
                self.discretizer.encode_returns(
                    np.asarray([beam.returns[index] for index in indices], np.float32)
                )
            )
            action_row = []
            reward_row = []
            done_row = []
            for index in indices:
                if index == 0:
                    action_row.append(
                        np.full(self.model.action_dim, self.discretizer.action_bins, np.int64)
                    )
                    reward_row.append(self.discretizer.reward_bins)
                    done_row.append(2)
                else:
                    action_row.append(self.discretizer.encode_actions(beam.actions[index - 1]))
                    reward_row.append(
                        int(self.discretizer.encode_rewards(
                            np.asarray([beam.rewards[index - 1]], np.float32)
                        )[0])
                    )
                    done_row.append(int(beam.dones[index - 1]))
            previous_actions.append(np.stack(action_row))
            previous_rewards.append(np.asarray(reward_row, np.int64))
            previous_dones.append(np.asarray(done_row, np.int64))
        states = torch.as_tensor(np.stack(state_rows), device=self.device)
        returns = torch.as_tensor(np.stack(return_rows), device=self.device)
        action_history = torch.as_tensor(np.stack(previous_actions), device=self.device)
        reward_history = torch.as_tensor(np.stack(previous_rewards), device=self.device)
        done_history = torch.as_tensor(np.stack(previous_dones), device=self.device)
        normalized_goal = (
            (np.asarray(goal, np.float32) - self.goal_mean) / self.goal_std
            if goal.size else np.empty(0, np.float32)
        )
        goals = torch.as_tensor(normalized_goal, device=self.device).view(1, 1, -1)
        goals = goals.expand(states.shape[0], states.shape[1], -1)
        valid = torch.ones(states.shape[:2], dtype=torch.bool, device=self.device)
        hidden = self.model.hidden(
            states, goals, returns, action_history, reward_history, done_history, valid
        )
        return hidden[:, -1]

    @torch.inference_mode()
    def act(
        self,
        current_state: np.ndarray,
        goal: np.ndarray,
        remaining_return: float,
        history: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        initial = _Beam(
            states=[np.asarray(item["state_tokens"], np.int64) for item in history]
            + [np.asarray(current_state, np.int64)],
            actions=[np.asarray(item["action"], np.float32) for item in history],
            rewards=[float(item["reward"]) for item in history],
            dones=[int(item["done"]) for item in history],
            returns=[float(item["return_to_go"]) for item in history]
            + [float(remaining_return)],
            score=0.0,
        )
        beams = [initial]
        temperature = max(float(self.config.get("temperature", 1.0)), 1e-4)
        beam_width = max(1, int(self.config.get("beam_width", 16)))
        action_candidates = max(1, int(self.config.get("action_candidates", 16)))
        action_top_k = max(1, int(self.config.get("action_top_k", 4)))
        action_weight = float(self.config.get("action_likelihood_weight", 0.05))
        transition_weight = float(self.config.get("transition_likelihood_weight", 0.01))
        reward_weight = float(self.config.get("reward_weight", 1.0))
        for _ in range(max(1, int(self.config.get("horizon", 5)))):
            # A predicted terminal beam keeps its accumulated score but is not
            # rolled into out-of-distribution post-terminal states.
            completed = [beam for beam in beams if beam.dones and beam.dones[-1]]
            active = [beam for beam in beams if not beam.dones or not beam.dones[-1]]
            if not active:
                break
            hidden = self._last_hidden(active, goal)
            action_log_probs = F.log_softmax(
                self.model.action_logits(hidden) / temperature, dim=-1
            )
            expanded: list[tuple[_Beam, int, np.ndarray]] = []
            for source_index, beam in enumerate(active):
                candidates = _top_factorized_actions(
                    action_log_probs[source_index], action_candidates, action_top_k
                )
                for action_tokens, log_likelihood in candidates:
                    normalized_action = np.asarray(
                        self.discretizer.decode_actions(action_tokens), np.float32
                    )
                    expanded.append((
                        _Beam(
                            states=list(beam.states), actions=list(beam.actions),
                            rewards=list(beam.rewards), dones=list(beam.dones),
                            returns=list(beam.returns),
                            score=beam.score + action_weight * log_likelihood,
                            first_action=(normalized_action.copy()
                                if beam.first_action is None else beam.first_action),
                        ),
                        source_index,
                        action_tokens,
                    ))
            expanded.sort(key=lambda item: item[0].score, reverse=True)
            expanded = expanded[:beam_width]
            source_indices = torch.as_tensor(
                [item[1] for item in expanded], device=self.device
            )
            candidate_tokens = torch.as_tensor(
                np.stack([item[2] for item in expanded]), device=self.device
            )
            transition = self.model.transition_logits(hidden[source_indices], candidate_tokens)
            next_tokens = transition["next_state"].argmax(-1)
            reward_tokens = transition["reward"].argmax(-1)
            done_tokens = transition["done"].argmax(-1)
            state_log_prob = F.log_softmax(transition["next_state"], dim=-1)
            selected_state_log_prob = state_log_prob.gather(
                -1, next_tokens.unsqueeze(-1)
            ).squeeze(-1).mean(-1)
            rewards = self.discretizer.decode_rewards(reward_tokens.float())
            next_beams = []
            for index, (beam, _, _) in enumerate(expanded):
                reward = float(rewards[index].item())
                done = int(done_tokens[index].item())
                action = np.asarray(
                    self.discretizer.decode_actions(candidate_tokens[index].detach().cpu().numpy()),
                    np.float32,
                )
                beam.actions.append(action)
                beam.rewards.append(reward)
                beam.dones.append(done)
                beam.states.append(next_tokens[index].detach().cpu().numpy())
                beam.returns.append(
                    (beam.returns[-1] - reward) / max(self.return_discount, 1e-6)
                )
                beam.score += (
                    reward_weight * reward
                    + transition_weight * float(selected_state_log_prob[index].item())
                )
                next_beams.append(beam)
            beams = completed + next_beams
            beams.sort(key=lambda item: item.score, reverse=True)
            beams = beams[:beam_width]
        best = max(beams, key=lambda item: item.score)
        if best.first_action is None:
            raise RuntimeError("Beam planner produced no action")
        return np.clip(best.first_action, -1.0, 1.0)


def _build_tokenizer(config: Mapping[str, Any]) -> VQVAE:
    return VQVAE(
        in_channels=int(config.get("in_channels", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        codebook_size=int(config.get("codebook_size", 512)),
        code_dim=int(config.get("code_dim", 128)),
        commitment_weight=float(config.get("commitment_weight", 0.25)),
    )


def evaluate_trajectory_transformer(
    checkpoint: str | Path,
    env: UnifiedControlEnv,
    episode_horizon: int,
    seeds: list[int],
    video_path: str | Path | None = None,
) -> list[dict[str, object]]:
    from tqdm.auto import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("algorithm") != "vq_trajectory_transformer":
        raise ValueError(
            "This checkpoint is the obsolete state-regression TT. Retrain TT with "
            "the VQ-token implementation before evaluation."
        )
    model = TrajectoryTransformer(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    tokenizer = _build_tokenizer(payload["tokenizer_config"]).to(device)
    tokenizer.load_state_dict(payload["tokenizer"])
    tokenizer.eval()
    statistics = {
        name: np.asarray(value, np.float32)
        for name, value in payload["statistics"].items()
    }
    discretizer = TrajectoryDiscretizer(**payload["discretizer"])
    planner = VQBeamPlanner(
        model, discretizer, payload["planner_config"],
        statistics["goal_mean"], statistics["goal_std"],
        float(payload.get("return_discount", 1.0)), device,
    )
    expected_shape = tuple(payload["token_shape"])
    results: list[dict[str, object]] = []
    episode_progress = tqdm(seeds, desc="TT beam-search evaluation", unit="episode")
    for episode_index, seed in enumerate(episode_progress):
        observation = env.reset(int(seed))
        remaining_return = float(statistics["target_return"])
        history: list[dict[str, Any]] = []
        frames = [env.render().copy()] if episode_index == 0 and video_path else []
        frame_metrics = [{"reward": 0.0, "success": False}] if frames else []
        rewards: list[float] = []
        success = False
        for step in range(int(episode_horizon)):
            with torch.inference_mode():
                token_grid = tokenizer.encode(
                    torch.as_tensor(
                        np.array(observation.rgb, copy=True), device=device
                    ).unsqueeze(0)
                )[0]
                if tuple(token_grid.shape) != expected_shape:
                    raise ValueError(
                        f"Live VQ token grid {tuple(token_grid.shape)} does not match "
                        f"training grid {expected_shape}"
                    )
                state_tokens = token_grid.reshape(-1).cpu().numpy()
            normalized_action = planner.act(
                state_tokens, np.asarray(observation.goal, np.float32),
                remaining_return, history,
            )
            action = env.action_low + 0.5 * (normalized_action + 1.0) * (
                env.action_high - env.action_low
            )
            transition = env.step(action)
            reward = float(transition.reward)
            success = success or bool(transition.info.get("success", False))
            history.append({
                "state_tokens": state_tokens,
                "action": normalized_action,
                "reward": reward,
                "done": int(transition.done),
                "return_to_go": remaining_return,
            })
            remaining_return = (remaining_return - reward) / max(
                float(payload.get("return_discount", 1.0)), 1e-6
            )
            rewards.append(reward)
            observation = transition.observation
            if frames:
                frames.append(env.render().copy())
                frame_metrics.append({"reward": reward, "success": success})
            if transition.done:
                break
        achieved = np.asarray(env.get_achieved_goal(), np.float32)
        desired = np.asarray(env.get_goal(), np.float32)
        results.append({
            "seed": int(seed),
            "return": float(sum(rewards)),
            "rewards": rewards,
            "success": success,
            "episode_length": step + 1,
            "final_goal_distance": (
                float(np.linalg.norm(achieved - desired))
                if achieved.size and achieved.shape == desired.shape else float("nan")
            ),
        })
        episode_progress.set_postfix(
            return_=f"{sum(rewards):.3g}", success=int(success), refresh=False
        )
        if frames:
            from fa_robotics_planner.visualization import save_rollout_video

            save_rollout_video(
                frames, frame_metrics, video_path,
                "VQ Trajectory Transformer + Beam Search", type(env).__name__,
                int(seed), int(episode_horizon),
            )
    return results
