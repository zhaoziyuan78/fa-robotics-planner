from __future__ import annotations

from typing import Any

import torch


def parameter_report(module: torch.nn.Module) -> dict[str, Any]:
    trainable = [(name, value) for name, value in module.named_parameters() if value.requires_grad]
    frozen = [(name, value) for name, value in module.named_parameters() if not value.requires_grad]
    return {
        "trainable_names": [name for name, _ in trainable],
        "trainable_parameters": sum(value.numel() for _, value in trainable),
        "frozen_names": [name for name, _ in frozen],
        "frozen_parameters": sum(value.numel() for _, value in frozen),
    }


def make_adapter_optimizer(
    method: torch.nn.Module,
    learning_rate: float = 1e-3,
    *,
    state_learning_rate: float | None = None,
    action_learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in method.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable adapter parameters")
    prior_parameters = {
        id(parameter)
        for name, child in method.named_children()
        if name in {"state_prior", "action_prior"}
        for parameter in child.parameters()
    }
    if any(id(parameter) in prior_parameters for parameter in parameters):
        raise AssertionError("Frozen prior parameter entered the adapter optimizer")
    adapter_groups = []
    grouped_ids: set[int] = set()
    for name, rate in (
        ("state_adapter", state_learning_rate),
        ("action_adapter", action_learning_rate),
    ):
        adapter = getattr(method, name, None)
        if adapter is None:
            continue
        selected = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
        if selected:
            adapter_groups.append(
                {
                    "params": selected,
                    "lr": float(learning_rate if rate is None else rate),
                }
            )
            grouped_ids.update(map(id, selected))
    remaining = [parameter for parameter in parameters if id(parameter) not in grouped_ids]
    if remaining:
        adapter_groups.append({"params": remaining, "lr": float(learning_rate)})
    return torch.optim.AdamW(adapter_groups, lr=float(learning_rate))
