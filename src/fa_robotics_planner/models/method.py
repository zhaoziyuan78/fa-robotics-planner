"""Composition root for the FunctionAlignmentWM method and ablations."""

from __future__ import annotations

import torch
from torch import nn

from .action_adapter import ActionAdapter
from .action_prior import CausalActionPrior
from .distributions import TanhNormal
from .state_adapter import StateAdapter
from .state_prior import CausalStatePrior
from .vqvae import VQVAE


class FunctionAlignmentWM(nn.Module):
    def __init__(
        self,
        action_prior: CausalActionPrior,
        state_prior: CausalStatePrior,
        state_adapter: StateAdapter | None,
        action_adapter: ActionAdapter | None,
        tokenizer: VQVAE | None = None,
        *,
        use_state_adapter: bool = True,
        use_action_adapter: bool = True,
    ):
        super().__init__()
        self.action_prior = action_prior
        self.state_prior = state_prior
        self.state_adapter = state_adapter
        self.action_adapter = action_adapter
        self.tokenizer = tokenizer
        self.use_state_adapter = bool(use_state_adapter)
        self.use_action_adapter = bool(use_action_adapter)
        if self.use_state_adapter and state_adapter is None:
            raise ValueError("State Adapter is enabled but no module was provided")
        if self.use_action_adapter and action_adapter is None:
            raise ValueError("Action Adapter is enabled but no module was provided")
        self.freeze_priors()
        self.apply_ablation()

    def freeze_priors(self) -> None:
        for module in (self.action_prior, self.state_prior, self.tokenizer):
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def apply_ablation(self) -> None:
        for enabled, adapter in (
            (self.use_state_adapter, self.state_adapter),
            (self.use_action_adapter, self.action_adapter),
        ):
            if adapter is not None:
                for parameter in adapter.parameters():
                    parameter.requires_grad = enabled

    def propose_actions(
        self,
        action_history: torch.Tensor,
        current_state: torch.Tensor,
        goal: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> TanhNormal:
        base, hidden = self.action_prior.next_distribution(action_history, valid_mask)
        return self.adapt_action_distribution(base, hidden, current_state, goal)

    def adapt_action_distribution(
        self,
        base: TanhNormal,
        prior_hidden: torch.Tensor,
        current_state: torch.Tensor,
        goal: torch.Tensor,
    ) -> TanhNormal:
        """Apply the optional adapter to a precomputed Action Prior output."""

        if not self.use_action_adapter:
            return base
        assert self.action_adapter is not None
        corrected, _, _ = self.action_adapter(base, prior_hidden, current_state, goal)
        return corrected

    def predict_transition(
        self,
        state_sequence: torch.Tensor,
        state_mask: torch.Tensor,
        current_action: torch.Tensor,
        video_tokens: torch.Tensor,
        valid_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.state_prior.encode_context(
            state_sequence, state_mask, video_tokens, valid_steps
        )
        if not self.use_state_adapter:
            tokens, _, _ = self.state_prior.generate_next_video(
                context.video_cache, context.observation_hidden
            )
            return context.passive_next, tokens
        assert self.state_adapter is not None
        predicted, _, condition = self.state_adapter(
            state_sequence[:, -1],
            context.passive_next,
            current_action,
            context.observation_hidden,
            context.video_summary,
        )

        def adapt(logits, token_hidden, spatial_index):
            return self.state_adapter.adapt_video_logits(
                logits, token_hidden, spatial_index, condition
            )[0]

        tokens, _, _ = self.state_prior.generate_next_video(
            context.video_cache,
            context.observation_hidden,
            logit_adapter=adapt,
        )
        return predicted, tokens

    def parameter_report(self) -> dict[str, object]:
        trainable = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        frozen = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        return {
            "trainable_names": trainable,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen_names": frozen,
            "frozen_parameters": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
        }
