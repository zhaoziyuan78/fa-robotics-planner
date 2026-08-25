"""Strictly action-only priors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .distributions import CategoricalActions, TanhNormal


@dataclass
class ActionPriorOutput:
    distribution: TanhNormal
    hidden: torch.Tensor


@dataclass(frozen=True)
class ActionPriorKVCache:
    """Per-layer normalized keys/values for autoregressive evaluation."""

    layer_inputs: tuple[torch.Tensor, ...]
    token_count: int


class CausalActionPrior(nn.Module):
    """Causal Transformer p(a_t | a_<t); no state/goal/task inputs exist."""

    def __init__(
        self,
        action_dim: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_length: int = 128,
        low: float | list[float] = -1.0,
        high: float | list[float] = 1.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_length = int(max_length)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * action_dim)
        self.register_buffer("low", torch.broadcast_to(torch.as_tensor(low, dtype=torch.float32), (action_dim,)).clone())
        self.register_buffer("high", torch.broadcast_to(torch.as_tensor(high, dtype=torch.float32), (action_dim,)).clone())

    def forward(self, action_history: torch.Tensor, valid_mask: torch.Tensor | None = None) -> ActionPriorOutput:
        if action_history.ndim != 3 or action_history.shape[-1] != self.action_dim:
            raise ValueError(f"action_history must be [B,T,{self.action_dim}]")
        batch, length, _ = action_history.shape
        if length >= self.max_length:
            action_history = action_history[:, -(self.max_length - 1) :]
            if valid_mask is not None:
                valid_mask = valid_mask[:, -(self.max_length - 1) :]
            length = action_history.shape[1]
        embedded = self.action_embed(action_history)
        tokens = torch.cat((self.start.expand(batch, -1, -1), embedded), dim=1)
        positions = self.position(torch.arange(length + 1, device=tokens.device))[None]
        tokens = tokens + positions
        causal = torch.triu(torch.ones(length + 1, length + 1, dtype=torch.bool, device=tokens.device), 1)
        padding = None
        if valid_mask is not None:
            if valid_mask.shape != (batch, length):
                raise ValueError("valid_mask must have shape [B,T]")
            padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device), ~valid_mask.bool()), 1)
        hidden = self.norm(self.transformer(tokens, mask=causal, src_key_padding_mask=padding))
        parameters = self.head(hidden)
        loc, log_scale = parameters.chunk(2, dim=-1)
        return ActionPriorOutput(TanhNormal(loc, log_scale, self.low, self.high), hidden)

    def next_distribution(self, action_history: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[TanhNormal, torch.Tensor]:
        output = self.forward(action_history, valid_mask)
        if valid_mask is None:
            index = torch.full((action_history.size(0),), output.hidden.size(1) - 1, device=action_history.device, dtype=torch.long)
        else:
            index = valid_mask.long().sum(dim=1)
        batch = torch.arange(action_history.size(0), device=action_history.device)
        return (
            TanhNormal(
                output.distribution.loc[batch, index],
                output.distribution.log_scale[batch, index],
                self.low,
                self.high,
            ),
            output.hidden[batch, index],
        )

    def build_kv_cache(
        self, action_history: torch.Tensor
    ) -> tuple[TanhNormal, torch.Tensor, ActionPriorKVCache]:
        """Encode a shared action prefix once and retain exact attention KV state.

        This path is evaluation-only. It is mathematically equivalent to
        ``next_distribution`` for an unpadded history, but exposes the
        per-layer prefix needed to append candidate actions without repeatedly
        re-encoding the common environment history.
        """

        if self.training:
            raise RuntimeError("Action Prior KV cache is only available in eval mode")
        if action_history.ndim != 3 or action_history.shape[-1] != self.action_dim:
            raise ValueError(f"action_history must be [B,T,{self.action_dim}]")
        if action_history.size(1) >= self.max_length:
            action_history = action_history[:, -(self.max_length - 1) :]
        batch, length, _ = action_history.shape
        embedded = self.action_embed(action_history)
        tokens = torch.cat((self.start.expand(batch, -1, -1), embedded), dim=1)
        tokens = tokens + self.position(
            torch.arange(length + 1, device=tokens.device)
        )[None]
        causal = torch.triu(
            torch.ones(length + 1, length + 1, dtype=torch.bool, device=tokens.device),
            1,
        )
        layer_inputs = []
        hidden = tokens
        for layer in self.transformer.layers:
            if not layer.norm_first:
                raise RuntimeError("KV cache requires norm_first Transformer layers")
            layer_inputs.append(layer.norm1(hidden))
            hidden = layer(hidden, src_mask=causal, is_causal=True)
        if self.transformer.norm is not None:
            hidden = self.transformer.norm(hidden)
        hidden = self.norm(hidden[:, -1])
        loc, log_scale = self.head(hidden).chunk(2, dim=-1)
        return (
            TanhNormal(loc, log_scale, self.low, self.high),
            hidden,
            ActionPriorKVCache(tuple(layer_inputs), length + 1),
        )

    def append_kv_cache(
        self, action: torch.Tensor, cache: ActionPriorKVCache
    ) -> tuple[TanhNormal, torch.Tensor, ActionPriorKVCache]:
        """Append one action token to an existing evaluation KV cache."""

        if self.training:
            raise RuntimeError("Action Prior KV cache is only available in eval mode")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must be [B,{self.action_dim}]")
        if cache.token_count >= self.max_length:
            raise ValueError("KV cache is full; rebuild it from the truncated action history")
        if len(cache.layer_inputs) != len(self.transformer.layers):
            raise ValueError("KV cache layer count does not match the Action Prior")

        hidden = self.action_embed(action) + self.position(
            torch.tensor(cache.token_count, device=action.device)
        )
        hidden = hidden[:, None]
        updated_inputs = []
        for layer, past in zip(self.transformer.layers, cache.layer_inputs):
            if past.size(0) == 1 and action.size(0) != 1:
                past = past.expand(action.size(0), -1, -1)
            elif past.size(0) != action.size(0):
                raise ValueError("KV cache batch size cannot be broadcast to the action batch")
            current = layer.norm1(hidden)
            key_value = torch.cat((past, current), dim=1)
            attention = layer.self_attn(
                current,
                key_value,
                key_value,
                need_weights=False,
                is_causal=False,
            )[0]
            hidden = hidden + layer.dropout1(attention)
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden))))
            )
            hidden = hidden + layer.dropout2(feed_forward)
            updated_inputs.append(key_value)
        if self.transformer.norm is not None:
            hidden = self.transformer.norm(hidden)
        hidden = self.norm(hidden[:, 0])
        loc, log_scale = self.head(hidden).chunk(2, dim=-1)
        return (
            TanhNormal(loc, log_scale, self.low, self.high),
            hidden,
            ActionPriorKVCache(tuple(updated_inputs), cache.token_count + 1),
        )


class RuleBasedActionPrior(nn.Module):
    """State-free OU/Gaussian prior used for data generation and smoke tests."""

    def __init__(self, action_dim: int, sigma: float = 0.35, smooth: float = 0.8):
        super().__init__()
        self.action_dim = int(action_dim)
        self.sigma = float(sigma)
        self.smooth = float(smooth)

    def next_distribution(self, action_history: torch.Tensor, valid_mask=None):
        if action_history.ndim != 3:
            raise ValueError("action_history must have shape [B,T,A]")
        if action_history.size(1):
            loc = torch.atanh((self.smooth * action_history[:, -1]).clamp(-0.999, 0.999))
        else:
            loc = action_history.new_zeros(action_history.size(0), self.action_dim)
        log_scale = torch.full_like(loc, float(torch.log(torch.tensor(self.sigma))))
        bounds = torch.ones(self.action_dim, device=loc.device)
        return TanhNormal(loc, log_scale, -bounds, bounds), loc


class DiscreteCausalActionPrior(nn.Module):
    """Strict action-token-only compatibility prior for discrete Windy runs."""

    def __init__(self, action_vocab: int, d_model: int = 256, n_layers: int = 2, dropout: float = 0.1, action_table=None):
        super().__init__()
        self.action_vocab = int(action_vocab)
        self.start_token = self.action_vocab
        self.embedding = nn.Embedding(self.action_vocab + 1, d_model)
        self.rnn = nn.GRU(
            d_model,
            d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.action_vocab)
        if action_table is None:
            self.action_table = None
        else:
            self.register_buffer("action_table", torch.as_tensor(action_table, dtype=torch.float32))

    def forward(self, action_tokens: torch.Tensor) -> tuple[CategoricalActions, torch.Tensor]:
        if action_tokens.ndim != 2:
            raise ValueError("action_tokens must have shape [B,T]")
        start = torch.full(
            (action_tokens.size(0), 1), self.start_token, dtype=torch.long, device=action_tokens.device
        )
        inputs = torch.cat((start, action_tokens), 1)
        hidden, _ = self.rnn(self.embedding(inputs))
        hidden = self.norm(hidden)
        return CategoricalActions(self.head(hidden), self.action_table), hidden

    def next_distribution(self, action_tokens: torch.Tensor) -> tuple[CategoricalActions, torch.Tensor]:
        distribution, hidden = self.forward(action_tokens)
        return CategoricalActions(distribution.logits[:, -1], self.action_table), hidden[:, -1]
