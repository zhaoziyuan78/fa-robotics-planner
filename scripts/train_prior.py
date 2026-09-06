from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.training.losses import action_prior_loss, masked_state_loss
from fa_robotics_planner.utils import seed_everything
from fa_robotics_planner.visualization import save_training_history

from ._common import checkpoint_path, config_from_unknown, data_path


def _translate(overrides: list[str]) -> list[str]:
    return [
        "env=" + item.split("=", 1)[1] if item.startswith("env_group=") else item
        for item in overrides
    ]


def _pad_action(batch):
    action_dim = batch[0]["actions"].shape[-1]
    length = max(int(item["sequence_length"]) for item in batch)
    actions = torch.zeros(len(batch), length, action_dim)
    valid = torch.zeros(len(batch), length, dtype=torch.bool)
    for index, item in enumerate(batch):
        size = int(item["sequence_length"])
        actions[index, :size] = torch.from_numpy(item["actions"][:size])
        valid[index, :size] = True
    return actions, valid


class _StateTokenDataset(Dataset):
    def __init__(self, observations: LazyEpisodeDataset, tokens: LazyEpisodeDataset):
        observation_ids = [int(item["id"]) for item in observations.entries]
        token_ids = [int(item["id"]) for item in tokens.entries]
        if observation_ids != token_ids:
            raise ValueError("Observation and token-cache episode IDs do not match")
        self.observations = observations
        self.tokens = tokens

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        observation = self.observations[index]
        token = self.tokens[index]
        return observation, token


def _pad_state(batch, context_frames: int, random_window: bool):
    maximum_frames = int(context_frames) + 1
    state_dim = batch[0][0]["control_state"].shape[-1]
    token_shape = batch[0][1]["video_tokens"].shape[1:]
    lengths = [
        min(
            int(observation["sequence_length"]),
            int(token["sequence_length"]),
            maximum_frames,
        )
        for observation, token in batch
    ]
    length = max(lengths)
    states = torch.zeros(len(batch), length, state_dim)
    masks = torch.zeros_like(states, dtype=torch.bool)
    tokens = torch.zeros(len(batch), length, *token_shape, dtype=torch.long)
    valid = torch.zeros(len(batch), length, dtype=torch.bool)
    for index, (observation, token) in enumerate(batch):
        total = min(
            int(observation["sequence_length"]), int(token["sequence_length"])
        )
        size = min(total, maximum_frames)
        start = (
            int(np.random.randint(0, total - size + 1))
            if random_window and total > size
            else 0
        )
        stop = start + size
        states[index, :size] = torch.from_numpy(
            observation["control_state"][start:stop]
        )
        masks[index, :size] = torch.from_numpy(
            observation["state_mask"][start:stop]
        )
        tokens[index, :size] = torch.from_numpy(token["video_tokens"][start:stop])
        valid[index, :size] = True
    return states, masks, tokens, valid


@dataclass
class _EarlyStopper:
    patience: int
    min_delta: float = 0.0
    warmup_epochs: int = 0
    best_loss: float = float("inf")
    best_epoch: int = 0
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early_stopping.patience must be positive")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("early_stopping.warmup_epochs must be non-negative")

    def step(self, loss: float, epoch: int) -> tuple[bool, bool]:
        if not np.isfinite(loss):
            raise FloatingPointError(f"Non-finite early-stopping loss: {loss}")
        improved = float(loss) < self.best_loss - self.min_delta
        if improved:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        elif int(epoch) >= self.warmup_epochs:
            self.stale_epochs += 1
        return improved, (
            int(epoch) >= self.warmup_epochs
            and self.stale_epochs >= self.patience
        )


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _observation_statistics(dataset: LazyEpisodeDataset) -> tuple[torch.Tensor, torch.Tensor]:
    first = dataset[0]
    state_dim = first["control_state"].shape[-1]
    total = np.zeros(state_dim, np.float64)
    square = np.zeros(state_dim, np.float64)
    count = np.zeros(state_dim, np.float64)
    for episode in dataset:
        state = np.asarray(episode["control_state"], np.float64)
        mask = np.asarray(episode["state_mask"], bool) & np.isfinite(state)
        values = np.where(mask, state, 0.0)
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += mask.sum(0)
    safe_count = np.maximum(count, 1.0)
    mean = total / safe_count
    variance = np.maximum(square / safe_count - np.square(mean), 1e-8)
    # Padding-only dimensions are left as identity-normalized zeros.
    mean[count == 0] = 0.0
    variance[count == 0] = 1.0
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(
        np.sqrt(variance).astype(np.float32)
    )


def _state_batch_losses(module, batch, device, phase: str, state_weight: float, video_weight: float):
    states, masks, tokens, valid = (value.to(device) for value in batch)
    transition_valid = valid[:, :-1] & valid[:, 1:]
    state_mask = masks[:, 1:] & transition_valid[..., None]
    zero = states.new_zeros(())
    if phase == "observation_warmup":
        passive = module.observation_warmup_forward(states, masks, valid)
        state_loss = masked_state_loss(passive, states[:, 1:], state_mask)
        return state_loss, state_loss, zero
    if phase == "video_warmup":
        logits = module.video_warmup_forward(tokens, valid)
        state_loss = zero
    else:
        output = module(states, masks, tokens, valid, cross_modal=True)
        state_loss = masked_state_loss(
            output.passive_next, states[:, 1:], state_mask
        )
        logits = output.video_logits
    targets = tokens[:, 1:].flatten(-2)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(*targets.shape)
    token_mask = transition_valid[..., None].expand_as(token_losses)
    video_loss = (token_losses * token_mask).sum() / token_mask.sum().clamp_min(1)
    if phase == "video_warmup":
        return video_loss, state_loss, video_loss
    return state_weight * state_loss + video_weight * video_loss, state_loss, video_loss


@torch.inference_mode()
def _evaluate_state(module, loader, device, phase, state_weight, video_weight):
    module.eval()
    totals = np.zeros(3, np.float64)
    for batch in loader:
        losses = _state_batch_losses(
            module, batch, device, phase, state_weight, video_weight
        )
        totals += np.asarray([loss.item() for loss in losses])
    return totals / max(1, len(loader))


def _phase_parameters(module, phase: str):
    if phase == "observation_warmup":
        children = (module.observation_prior, module.observation_head)
    elif phase == "video_warmup":
        children = (module.video_prior,)
    else:
        children = (module,)
    selected = {id(parameter) for child in children for parameter in child.parameters()}
    for parameter in module.parameters():
        parameter.requires_grad = id(parameter) in selected
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _train_state_prior(method, train, validation, config, device):
    module = method.state_prior
    mean, std = _observation_statistics(train.observations)
    module.set_observation_stats(mean.to(device), std.to(device))
    state_config = config["model"]["state_prior"]
    training_config = state_config.get("training", {})
    context_frames = int(state_config.get("context_frames", 8))
    batch_size = int(config.get("batch_size", state_config.get("batch_size", 4)))
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            _pad_state, context_frames=context_frames, random_window=True
        ),
    )
    validation_loader = (
        DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(
                _pad_state, context_frames=context_frames, random_window=False
            ),
        )
        if len(validation)
        else None
    )
    phases = [
        ("observation_warmup", int(training_config.get("observation_warmup_epochs", 5))),
        ("video_warmup", int(training_config.get("video_warmup_epochs", 5))),
        ("joint", int(config.get("epochs", training_config.get("joint_epochs", 10)))),
    ]
    state_weight = float(training_config.get("state_loss_weight", 1.0))
    video_weight = float(training_config.get("video_loss_weight", 1.0))
    gradient_clip = float(state_config.get("gradient_clip_norm", 1.0))
    learning_rate = float(state_config.get("learning_rate", 2e-4))
    scheduler_config = state_config.get("scheduler", {})
    early_config = state_config.get("early_stopping", {})
    history = {"state_prior": [], "state": [], "video_token": [], "learning_rate": [], "phase": []}
    phase_summaries = []
    for phase, epochs in phases:
        if epochs <= 0:
            continue
        parameters = _phase_parameters(module, phase)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        scheduler = None
        if str(scheduler_config.get("name", "none")).lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 8)),
                threshold=float(scheduler_config.get("threshold", 1e-3)),
                threshold_mode="abs",
                cooldown=int(scheduler_config.get("cooldown", 0)),
                min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            )
        stopper = (
            _EarlyStopper(
                int(early_config.get("patience", 30)),
                float(early_config.get("min_delta", 1e-3)),
                int(early_config.get("warmup_epochs", 20)),
            )
            if phase == "joint" and bool(early_config.get("enabled", False))
            else None
        )
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        completed = 0
        early_stopped = False
        for epoch in range(epochs):
            module.train()
            totals = np.zeros(3, np.float64)
            for batch in train_loader:
                losses = _state_batch_losses(
                    module, batch, device, phase, state_weight, video_weight
                )
                if not torch.isfinite(losses[0]):
                    raise FloatingPointError(
                        f"Non-finite {phase} loss: state={losses[1].item()} video={losses[2].item()}"
                    )
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters, gradient_clip, error_if_nonfinite=True
                    )
                optimizer.step()
                totals += np.asarray([loss.item() for loss in losses])
            averages = totals / max(1, len(train_loader))
            monitor = (
                _evaluate_state(
                    module, validation_loader, device, phase, state_weight, video_weight
                )[0]
                if validation_loader is not None
                else averages[0]
            )
            should_stop = False
            if stopper is not None:
                improved, should_stop = stopper.step(float(monitor), epoch + 1)
            else:
                improved = monitor < best_loss
            if improved:
                best_loss = float(monitor)
                best_epoch = epoch + 1
                best_state = _cpu_state_dict(module)
            if scheduler is not None:
                scheduler.step(float(monitor))
            history["state_prior"].append(float(averages[0]))
            history["state"].append(float(averages[1]))
            history["video_token"].append(float(averages[2]))
            history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            history["phase"].append(phase)
            completed = epoch + 1
            print(
                f"phase={phase} epoch={epoch + 1}/{epochs} loss={averages[0]:.6f} "
                f"state={averages[1]:.6f} video={averages[2]:.6f} monitor={monitor:.6f}"
            )
            if should_stop:
                early_stopped = True
                print(f"Early stopping {phase} at epoch {epoch + 1}")
                break
        if best_state is not None:
            module.load_state_dict(best_state)
        phase_summaries.append(
            {
                "phase": phase,
                "epochs_requested": epochs,
                "epochs_completed": completed,
                "early_stopped": early_stopped,
                "best_epoch": best_epoch,
                "best_monitor_loss": best_loss,
                "restored_best_weights": best_state is not None,
                "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    for parameter in module.parameters():
        parameter.requires_grad = False
    return {
        "state_prior": module.state_dict(),
        "tokenizer": method.tokenizer.state_dict(),
    }, history, {"phases": phase_summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=("state", "action"))
    parser.add_argument("--data")
    parser.add_argument("--tokens")
    parser.add_argument("--vqvae")
    parser.add_argument("--output")
    args, unknown = parser.parse_known_args()
    overrides = _translate(unknown)
    prior = args.prior or next(
        (item.split("=", 1)[1] for item in overrides if item.startswith("prior=")),
        None,
    )
    if prior not in {"state", "action"}:
        raise ValueError("Specify prior=state or prior=action")
    overrides = [item for item in overrides if not item.startswith("prior=")]
    config = config_from_unknown(["model=prior_adapter", *overrides])
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    env_name = config["env"]["name"]
    data = Path(args.data or config.get("data", data_path(config, env_name, f"{prior}_prior")))
    raw_train = LazyEpisodeDataset(
        data,
        split="train",
        seed=seed,
        max_transitions=config.get("max_transitions"),
        min_sequence_length=2 if prior == "state" else 1,
    )
    if not len(raw_train):
        raise ValueError(f"No usable {prior} training episodes in {data}")
    sample = raw_train[0]
    method = build_method(config, sample.get("action_low"), sample.get("action_high"))
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    method.to(device)
    started = time.perf_counter()
    if prior == "action":
        module = method.action_prior
        for parameter in module.parameters():
            parameter.requires_grad = True
        batch_size = int(config.get("batch_size", 64))
        loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True, collate_fn=_pad_action)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(config["model"]["action_prior"].get("learning_rate", 2e-4)),
        )
        epochs = int(config.get("epochs", 20))
        history = {"action_prior": []}
        for epoch in range(epochs):
            total = 0.0
            for actions, valid in loader:
                actions, valid = actions.to(device), valid.to(device)
                output = module(actions[:, :-1], valid[:, :-1])
                loss = action_prior_loss(output.distribution, actions, valid)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
            average = total / max(1, len(loader))
            history["action_prior"].append(average)
            print(f"epoch={epoch + 1} action_prior_loss={average:.6f}")
        state = module.state_dict()
        training_summary: Mapping[str, Any] = {"epochs_completed": epochs}
        architecture_version = 1
    else:
        token_root = Path(args.tokens or data_path(config, env_name, "tokens/state_prior"))
        token_train = LazyEpisodeDataset(
            token_root,
            split="train",
            seed=seed,
            max_transitions=config.get("max_transitions"),
            min_sequence_length=2,
        )
        raw_validation = LazyEpisodeDataset(
            data,
            split="val",
            seed=seed,
            max_transitions=config.get("max_transitions"),
            min_sequence_length=2,
        )
        token_validation = LazyEpisodeDataset(
            token_root,
            split="val",
            seed=seed,
            max_transitions=config.get("max_transitions"),
            min_sequence_length=2,
        )
        train = _StateTokenDataset(raw_train, token_train)
        validation = _StateTokenDataset(raw_validation, token_validation)
        tokenizer_checkpoint = Path(
            args.vqvae
            or config.get(
                "vqvae_checkpoint",
                checkpoint_path(config, "tokenizers", f"{env_name}_vqvae_seed{seed}.pt"),
            )
        )
        loaded = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
        if loaded.get("kind") != "vqvae":
            raise ValueError(f"Not a VQ-VAE checkpoint: {tokenizer_checkpoint}")
        method.tokenizer.load_state_dict(loaded["tokenizer"])
        method.tokenizer.eval()
        for parameter in method.tokenizer.parameters():
            parameter.requires_grad = False
        state, history, training_summary = _train_state_prior(
            method, train, validation, config, device
        )
        architecture_version = method.state_prior.architecture_version
    output = Path(
        args.output
        or config.get(
            "output",
            checkpoint_path(config, "priors", f"{env_name}_{prior}_prior.pt"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": f"{prior}_prior",
            "architecture_version": architecture_version,
            "config": config,
            "state": state,
            "wall_clock_train_seconds": time.perf_counter() - started,
            "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "history": history,
            "training_summary": training_summary,
        },
        output,
    )
    numeric_history = {
        name: values
        for name, values in history.items()
        if values and isinstance(values[0], (int, float, np.number))
    }
    save_training_history(
        numeric_history,
        output.parent / f"{output.stem}_diagnostics" / "training_loss",
    )
    print(f"Saved {prior} prior to {output}")


if __name__ == "__main__":
    main()
