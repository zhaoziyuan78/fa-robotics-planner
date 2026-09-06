"""Coupled autoregressive video-token and structured-observation priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StatePriorOutput:
    """Teacher-forced predictions for every adjacent frame pair."""

    passive_next: torch.Tensor
    observation_hidden: torch.Tensor
    video_logits: torch.Tensor
    video_predictor_hidden: torch.Tensor
    video_hidden: torch.Tensor
    video_summary: torch.Tensor

    @property
    def hidden(self) -> torch.Tensor:
        """Observation hidden aligned with ``passive_next`` transitions."""

        return self.observation_hidden[:, :-1]


@dataclass
class StatePriorContext:
    passive_next: torch.Tensor
    observation_hidden: torch.Tensor
    video_summary: torch.Tensor
    video_cache: dict[str, object]


class ObservationAutoregressivePrior(nn.Module):
    """Causal observation-only branch; no proprioception enters this module."""

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        model_type: str,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_frames: int,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.d_model = int(d_model)
        self.model_type = str(model_type)
        self.input = nn.Linear(2 * self.state_dim, self.d_model)
        self.position = nn.Embedding(int(max_frames), self.d_model)
        if self.model_type == "gru":
            self.backbone = nn.GRU(
                self.d_model,
                self.d_model,
                num_layers=int(n_layers),
                dropout=float(dropout) if int(n_layers) > 1 else 0.0,
                batch_first=True,
            )
        elif self.model_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                self.d_model,
                int(n_heads),
                4 * self.d_model,
                float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, int(n_layers))
        else:
            raise ValueError("observation model_type must be 'gru' or 'transformer'")
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        states: torch.Tensor,
        state_mask: torch.Tensor,
        valid_steps: torch.Tensor | None,
    ) -> torch.Tensor:
        masked = states * state_mask.to(states.dtype)
        hidden = self.input(
            torch.cat((masked, state_mask.to(states.dtype)), dim=-1)
        )
        positions = torch.arange(states.size(1), device=states.device).clamp_max(
            self.position.num_embeddings - 1
        )
        hidden = hidden + self.position(positions)[None]
        if self.model_type == "gru":
            hidden, _ = self.backbone(hidden)
        else:
            length = states.size(1)
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=states.device), 1
            )
            padding = ~valid_steps.bool() if valid_steps is not None else None
            hidden = self.backbone(
                hidden, mask=causal, src_key_padding_mask=padding
            )
        return self.norm(hidden)


class VideoAutoregressivePrior(nn.Module):
    """Raster-order causal Transformer over discrete frame tokens only."""

    def __init__(
        self,
        codebook_size: int,
        tokens_per_frame: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_frames: int,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.tokens_per_frame = int(tokens_per_frame)
        self.d_model = int(d_model)
        self.max_frames = int(max_frames)
        self.token_embedding = nn.Embedding(self.codebook_size, self.d_model)
        self.frame_embedding = nn.Embedding(self.max_frames, self.d_model)
        self.spatial_embedding = nn.Embedding(self.tokens_per_frame, self.d_model)
        self.dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            self.d_model,
            int(n_heads),
            int(d_ff),
            float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.codebook_size)

    def _flatten(self, tokens: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if tokens.ndim == 4:
            tokens = tokens.flatten(-2)
        if tokens.ndim != 3:
            raise ValueError("video_tokens must have shape [B,T,N] or [B,T,H,W]")
        batch, frames, count = tokens.shape
        if count != self.tokens_per_frame:
            raise ValueError(
                f"Expected {self.tokens_per_frame} video tokens per frame, got {count}"
            )
        return tokens.long().reshape(batch, frames * count), frames, count

    def _embeddings(self, token_ids: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        positions = torch.arange(
            int(start_position),
            int(start_position) + token_ids.size(1),
            device=token_ids.device,
        )
        frame_ids = torch.div(
            positions, self.tokens_per_frame, rounding_mode="floor"
        ).clamp_max(self.max_frames - 1)
        spatial_ids = positions % self.tokens_per_frame
        return self.dropout(
            self.token_embedding(token_ids.long())
            + self.frame_embedding(frame_ids)[None]
            + self.spatial_embedding(spatial_ids)[None]
        )

    def forward(
        self,
        video_tokens: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat, frames, count = self._flatten(video_tokens)
        hidden = self._embeddings(flat)
        length = hidden.size(1)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=hidden.device), 1
        )
        padding = None
        if valid_steps is not None:
            if valid_steps.shape != (flat.size(0), frames):
                raise ValueError("valid_steps must match video frame dimensions")
            padding = ~valid_steps.bool().repeat_interleave(count, dim=1)
        return self.norm(
            self.transformer(hidden, mask=causal, src_key_padding_mask=padding)
        ).reshape(flat.size(0), frames, count, self.d_model)

    @staticmethod
    def _shape_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, width = value.shape
        return value.view(batch, length, heads, width // heads).transpose(1, 2)

    @staticmethod
    def _project_qkv(attention, query: torch.Tensor, key_value: torch.Tensor):
        width = attention.embed_dim
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        # Incremental self-attention always projects the same normalized token
        # as Q, K and V.  Use the packed MultiheadAttention projection in one
        # GEMM instead of launching three small Linear kernels per layer/token.
        if query is key_value:
            return F.linear(query, weight, bias).chunk(3, dim=-1)
        q_bias = bias[:width] if bias is not None else None
        k_bias = bias[width : 2 * width] if bias is not None else None
        v_bias = bias[2 * width :] if bias is not None else None
        return (
            F.linear(query, weight[:width], q_bias),
            F.linear(key_value, weight[width : 2 * width], k_bias),
            F.linear(key_value, weight[2 * width :], v_bias),
        )

    def _cached_attention(
        self, layer, query, key_value, previous, attention_mask
    ):
        attention = layer.self_attn
        query, key, value = self._project_qkv(attention, query, key_value)
        query = self._shape_heads(query, attention.num_heads)
        key = self._shape_heads(key, attention.num_heads)
        value = self._shape_heads(value, attention.num_heads)
        if previous is not None and "used" in previous:
            used = int(previous["used"])
            end = used + key.size(2)
            if end > previous["key"].size(2):
                raise RuntimeError("Reserved video K/V cache capacity was exceeded")
            previous["key"][:, :, used:end].copy_(key)
            previous["value"][:, :, used:end].copy_(value)
            key = previous["key"][:, :, :end]
            value = previous["value"][:, :, :end]
            next_cache = {
                "key": previous["key"],
                "value": previous["value"],
                "used": end,
            }
        else:
            if previous is not None:
                key = torch.cat((previous["key"], key), dim=2)
                value = torch.cat((previous["value"], value), dim=2)
            next_cache = {"key": key, "value": value}
        query = query * ((attention.embed_dim // attention.num_heads) ** -0.5)
        scores = query @ key.transpose(-2, -1)
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask[None, None], torch.finfo(scores.dtype).min
            )
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = F.dropout(
            probabilities, p=attention.dropout, training=attention.training
        )
        result = probabilities @ value
        result = result.transpose(1, 2).contiguous().view(
            result.size(0), result.size(2), attention.embed_dim
        )
        return attention.out_proj(result), next_cache

    def _incremental_layer(self, layer, hidden, previous, attention_mask):
        if layer.norm_first:
            normalized = layer.norm1(hidden)
            attended, cache = self._cached_attention(
                layer, normalized, normalized, previous, attention_mask
            )
            hidden = hidden + layer.dropout1(attended)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden))))
            )
            return hidden + layer.dropout2(feedforward), cache
        attended, cache = self._cached_attention(
            layer, hidden, hidden, previous, attention_mask
        )
        hidden = layer.norm1(hidden + layer.dropout1(attended))
        feedforward = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(hidden)))
        )
        return layer.norm2(hidden + layer.dropout2(feedforward)), cache

    @torch.no_grad()
    def append_to_cache(self, cache: dict[str, object] | None, tokens: torch.Tensor):
        if tokens.ndim == 1:
            tokens = tokens[:, None]
        start = 0 if cache is None else int(cache["length"])
        hidden = self._embeddings(tokens.long(), start)
        previous_length = start
        # A single token appended to a cache may attend to the entire prefix
        # and itself.  Constructing an all-False mask here used to allocate a
        # sequence-length tensor and launch a redundant masked_fill kernel in
        # every Transformer layer for every generated raster token.
        mask = None
        if cache is None or tokens.size(1) > 1:
            query_positions = torch.arange(
                tokens.size(1), device=tokens.device
            )[:, None]
            key_positions = torch.arange(
                previous_length + tokens.size(1), device=tokens.device
            )[None]
            mask = key_positions > previous_length + query_positions
        layers = []
        for index, layer in enumerate(self.transformer.layers):
            previous = None if cache is None else cache["layers"][index]
            hidden, next_layer = self._incremental_layer(
                layer, hidden, previous, mask
            )
            layers.append(next_layer)
        hidden = self.norm(hidden)
        next_cache: dict[str, object] = {
            "layers": layers,
            "length": previous_length + tokens.size(1),
            "last_hidden": hidden[:, -1],
        }
        if cache is None:
            return next_cache, hidden
        cache.update(next_cache)
        return cache, hidden

    @torch.no_grad()
    def build_cache(self, video_tokens: torch.Tensor) -> dict[str, object]:
        flat, _, _ = self._flatten(video_tokens)
        cache, hidden = self.append_to_cache(None, flat)
        cache["last_frame_summary"] = hidden[:, -self.tokens_per_frame :].mean(1)
        return cache

    @staticmethod
    @torch.no_grad()
    def clone_cache(cache: dict[str, object]) -> dict[str, object]:
        return {
            "layers": [
                {"key": item["key"].clone(), "value": item["value"].clone()}
                for item in cache["layers"]
            ],
            "length": int(cache["length"]),
            "last_hidden": cache["last_hidden"].clone(),
            "last_frame_summary": cache["last_frame_summary"].clone(),
        }

    @staticmethod
    @torch.no_grad()
    def repeat_cache(
        cache: dict[str, object],
        batch_size: int,
        additional_tokens: int = 0,
    ) -> dict[str, object]:
        """Expand a shared prefix, optionally reserving an in-place suffix.

        Autoregressive generation previously concatenated the complete prefix
        K/V tensor for every new raster token.  A reserved cache copies the
        common prefix once and writes projected suffix keys/values in place.
        """

        batch_size = int(batch_size)
        additional_tokens = int(additional_tokens)
        if batch_size < 1 or additional_tokens < 0:
            raise ValueError(
                "Cache batch size must be positive and capacity non-negative"
            )

        def repeat(value: torch.Tensor) -> torch.Tensor:
            if value.size(0) == batch_size:
                return value.clone()
            if value.size(0) != 1:
                raise ValueError("Only a batch-one video cache can be expanded")
            return value.expand(batch_size, *value.shape[1:]).clone()

        def reserve(value: torch.Tensor) -> torch.Tensor:
            if value.size(0) == batch_size:
                repeated = value
            elif value.size(0) == 1:
                repeated = value.expand(batch_size, *value.shape[1:])
            else:
                raise ValueError("Only a batch-one video cache can be expanded")
            capacity = value.size(2) + additional_tokens
            result = value.new_empty(
                batch_size, value.size(1), capacity, value.size(3)
            )
            result[:, :, : value.size(2)].copy_(repeated)
            return result

        if additional_tokens:
            layers = [
                {
                    "key": reserve(item["key"]),
                    "value": reserve(item["value"]),
                    "used": int(cache["length"]),
                }
                for item in cache["layers"]
            ]
        else:
            layers = [
                {"key": repeat(item["key"]), "value": repeat(item["value"])}
                for item in cache["layers"]
            ]
        return {
            "layers": layers,
            "length": int(cache["length"]),
            "last_hidden": repeat(cache["last_hidden"]),
            "last_frame_summary": repeat(cache["last_frame_summary"]),
        }


class CausalStatePrior(nn.Module):
    """Two causal priors joined only by directional conditioning MLPs."""

    architecture_version = 2

    def __init__(
        self,
        state_dim: int,
        *,
        codebook_size: int = 512,
        tokens_per_frame: int = 64,
        video_d_model: int = 256,
        video_layers: int = 4,
        video_heads: int = 8,
        video_d_ff: int = 1024,
        observation_type: str = "auto",
        observation_d_model: int | None = None,
        observation_layers: int = 2,
        observation_heads: int = 4,
        dropout: float = 0.1,
        context_frames: int = 8,
        max_frames: int = 128,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        max_length: int | None = None,
        **forbidden,
    ):
        super().__init__()
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise TypeError(f"Unsupported State Prior arguments: {names}")
        self.state_dim = int(state_dim)
        self.context_frames = int(
            context_frames if max_length is None else min(context_frames, max_length)
        )
        self.max_length = self.context_frames
        self.tokens_per_frame = int(tokens_per_frame)
        self.codebook_size = int(codebook_size)
        if observation_type == "auto":
            observation_type = "gru" if self.state_dim <= 16 else "transformer"
        if observation_d_model is None:
            observation_d_model = int(
                d_model or (128 if self.state_dim <= 64 else 256)
            )
        if n_layers is not None:
            observation_layers = int(n_layers)
        if n_heads is not None:
            observation_heads = int(n_heads)
        self.observation_prior = ObservationAutoregressivePrior(
            self.state_dim,
            int(observation_d_model),
            observation_type,
            int(observation_layers),
            int(observation_heads),
            float(dropout),
            int(max_frames),
        )
        self.video_prior = VideoAutoregressivePrior(
            self.codebook_size,
            self.tokens_per_frame,
            int(video_d_model),
            int(video_layers),
            int(video_heads),
            int(video_d_ff),
            float(dropout),
            int(max_frames),
        )
        obs_dim = self.observation_prior.d_model
        video_dim = self.video_prior.d_model
        self.video_to_observation = nn.Sequential(
            nn.Linear(video_dim, obs_dim), nn.SiLU(), nn.Linear(obs_dim, obs_dim)
        )
        self.observation_to_video = nn.Sequential(
            nn.Linear(obs_dim, video_dim), nn.SiLU(), nn.Linear(video_dim, video_dim)
        )
        self.observation_head = nn.Linear(obs_dim, self.state_dim)
        for module in (
            self.video_to_observation[-1],
            self.observation_to_video[-1],
            self.observation_head,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.register_buffer("observation_mean", torch.zeros(self.state_dim))
        self.register_buffer("observation_std", torch.ones(self.state_dim))

    @property
    def observation_hidden_dim(self) -> int:
        return self.observation_prior.d_model

    @property
    def video_hidden_dim(self) -> int:
        return self.video_prior.d_model

    def set_observation_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.state_dim or std.numel() != self.state_dim:
            raise ValueError("Observation normalization statistics have wrong size")
        self.observation_mean.copy_(mean.reshape(-1).to(self.observation_mean))
        self.observation_std.copy_(
            std.reshape(-1).clamp_min(1e-6).to(self.observation_std)
        )

    def _normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.observation_mean) / self.observation_std

    def _predict_state(
        self,
        states: torch.Tensor,
        observation_hidden: torch.Tensor,
        video_summary: torch.Tensor,
        cross_modal: bool,
    ) -> torch.Tensor:
        fused = observation_hidden
        if cross_modal:
            fused = fused + self.video_to_observation(video_summary)
        normalized_current = self._normalize(states)
        normalized_next = normalized_current + self.observation_head(fused)
        return normalized_next * self.observation_std + self.observation_mean

    def _condition_video(
        self,
        predictor_hidden: torch.Tensor,
        observation_hidden: torch.Tensor,
        cross_modal: bool,
    ) -> torch.Tensor:
        if cross_modal:
            predictor_hidden = predictor_hidden + self.observation_to_video(
                observation_hidden
            )
        return self.video_prior.head(predictor_hidden)

    def observation_warmup_forward(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self._normalize(state_sequence) * state_mask.to(
            state_sequence.dtype
        )
        hidden = self.observation_prior(normalized, state_mask, valid_steps)
        return self._predict_state(
            state_sequence[:, :-1],
            hidden[:, :-1],
            hidden.new_zeros(
                (*hidden[:, :-1].shape[:-1], self.video_hidden_dim)
            ),
            False,
        )

    def video_warmup_forward(
        self,
        video_tokens: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.video_prior(video_tokens, valid_steps)
        batch, frames, count, width = hidden.shape
        flat = hidden.reshape(batch, frames * count, width)
        predictors = torch.stack(
            [
                flat[:, frame * count - 1 : frame * count + count - 1]
                for frame in range(1, frames)
            ],
            dim=1,
        )
        return self.video_prior.head(predictors)

    def forward(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
        *,
        cross_modal: bool = True,
    ) -> StatePriorOutput:
        if state_sequence.ndim != 3 or state_sequence.size(-1) != self.state_dim:
            raise ValueError(f"state_sequence must be [B,T,{self.state_dim}]")
        if state_mask.shape != state_sequence.shape:
            raise ValueError("state_mask must match state_sequence")
        if video_tokens.shape[:2] != state_sequence.shape[:2]:
            raise ValueError("video_tokens and state_sequence must share [B,T]")
        if state_sequence.size(1) < 2:
            raise ValueError("State Prior training requires at least two frames")
        normalized = self._normalize(state_sequence) * state_mask.to(
            state_sequence.dtype
        )
        observation_hidden = self.observation_prior(
            normalized, state_mask, valid_steps
        )
        video_hidden = self.video_prior(video_tokens, valid_steps)
        video_summary = video_hidden.mean(dim=2)
        passive_next = self._predict_state(
            state_sequence[:, :-1],
            observation_hidden[:, :-1],
            video_summary[:, :-1],
            cross_modal,
        )
        batch, frames, count, width = video_hidden.shape
        flat_hidden = video_hidden.reshape(batch, frames * count, width)
        predictors = torch.stack(
            [
                flat_hidden[:, frame * count - 1 : frame * count + count - 1]
                for frame in range(1, frames)
            ],
            dim=1,
        )
        observation_condition = observation_hidden[:, :-1, None].expand(
            -1, -1, count, -1
        )
        video_logits = self._condition_video(
            predictors, observation_condition, cross_modal
        )
        return StatePriorOutput(
            passive_next,
            observation_hidden,
            video_logits,
            predictors,
            video_hidden,
            video_summary,
        )

    def encode_context(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
        *,
        cross_modal: bool = True,
    ) -> StatePriorContext:
        # Building the projected K/V cache already computes the exact causal
        # hidden states.  Re-running ``video_prior.forward`` here used to encode
        # the same (potentially 8 x 144-token) history a second time.
        video_cache = self.video_prior.build_cache(video_tokens)
        video_summary = video_cache["last_frame_summary"]
        passive, observation_hidden = self.encode_observation_context(
            state_sequence,
            state_mask,
            video_summary,
            valid_steps,
            cross_modal=cross_modal,
        )
        return StatePriorContext(
            passive,
            observation_hidden,
            video_summary,
            video_cache,
        )

    def encode_observation_context(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        video_summary: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
        *,
        cross_modal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance observation dynamics using an already cached video summary."""

        normalized = self._normalize(state_sequence) * state_mask.to(
            state_sequence.dtype
        )
        observation_hidden = self.observation_prior(
            normalized, state_mask, valid_steps
        )[:, -1]
        passive = self._predict_state(
            state_sequence[:, -1], observation_hidden, video_summary, cross_modal
        )
        return passive, observation_hidden

    @torch.no_grad()
    def generate_next_video(
        self,
        cache: dict[str, object],
        observation_hidden: torch.Tensor,
        *,
        cross_modal: bool = True,
        logit_adapter: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
        | None = None,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        generated = []
        generated_hidden = []
        # Observation context is fixed while decoding one frame.  Project it
        # once instead of repeating the same two-layer MLP 144 times on Fetch.
        observation_condition = (
            self.observation_to_video(observation_hidden) if cross_modal else None
        )
        for spatial_index in range(self.tokens_per_frame):
            token_hidden = cache["last_hidden"]
            conditioned_hidden = (
                token_hidden
                if observation_condition is None
                else token_hidden + observation_condition
            )
            logits = self.video_prior.head(conditioned_hidden)
            if logit_adapter is not None:
                logits = logit_adapter(logits, token_hidden, spatial_index)
            token = (
                torch.distributions.Categorical(logits=logits).sample()
                if sample
                else logits.argmax(dim=-1)
            )
            generated.append(token)
            cache, hidden = self.video_prior.append_to_cache(cache, token)
            generated_hidden.append(hidden[:, -1])
        summary = torch.stack(generated_hidden, dim=1).mean(1)
        cache["last_frame_summary"] = summary
        return torch.stack(generated, dim=1), summary, cache
