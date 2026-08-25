from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.training.losses import action_prior_loss, masked_state_loss
from fa_robotics_planner.utils import seed_everything
from fa_robotics_planner.visualization import save_training_history

from ._common import checkpoint_path, config_from_unknown, data_path


def _translate(overrides: list[str]) -> list[str]:
    translated = []
    for item in overrides:
        if item.startswith("env_group="):
            translated.append("env=" + item.split("=", 1)[1])
        else:
            translated.append(item)
    return translated


def _pad_action(batch):
    action_dim = batch[0]["actions"].shape[-1]
    length = max(int(item["sequence_length"]) for item in batch)
    actions = torch.zeros(len(batch), length, action_dim)
    valid = torch.zeros(len(batch), length, dtype=torch.bool)
    for index, item in enumerate(batch):
        size = int(item["sequence_length"])
        actions[index, :size] = torch.from_numpy(item["actions"])
        valid[index, :size] = True
    return actions, valid


def _pad_state(batch):
    length = max(int(item["sequence_length"]) for item in batch) - 1
    state_dim = batch[0]["control_state"].shape[-1]
    proprio_dim = batch[0]["proprio"].shape[-1]
    image_shape = batch[0]["rgb"].shape[1:]
    states = torch.zeros(len(batch), length, state_dim)
    targets = torch.zeros_like(states)
    masks = torch.zeros_like(states, dtype=torch.bool)
    proprio = torch.zeros(len(batch), length, proprio_dim)
    target_proprio = torch.zeros_like(proprio)
    rgb = torch.zeros(len(batch), length, *image_shape, dtype=torch.uint8)
    target_rgb = torch.zeros_like(rgb)
    valid = torch.zeros(len(batch), length, dtype=torch.bool)
    for index, item in enumerate(batch):
        size = max(0, int(item["sequence_length"]) - 1)
        states[index, :size] = torch.from_numpy(item["control_state"][:size])
        targets[index, :size] = torch.from_numpy(item["control_state"][1 : size + 1])
        masks[index, :size] = torch.from_numpy(item["state_mask"][1 : size + 1])
        proprio[index, :size] = torch.from_numpy(item["proprio"][:size])
        target_proprio[index, :size] = torch.from_numpy(item["proprio"][1 : size + 1])
        rgb[index, :size] = torch.from_numpy(item["rgb"][:size])
        target_rgb[index, :size] = torch.from_numpy(item["rgb"][1 : size + 1])
        valid[index, :size] = True
    return states, targets, masks, proprio, target_proprio, rgb, target_rgb, valid


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
        should_stop = (
            int(epoch) >= self.warmup_epochs
            and self.stale_epochs >= self.patience
        )
        return improved, should_stop


def _state_prior_batch_losses(
    method,
    module,
    batch,
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states, targets, masks, proprio, target_proprio, rgb, target_rgb, valid = (
        value.to(device) for value in batch
    )
    visual = method.visual_encoder(rgb) if method.visual_encoder is not None else None
    output = module(
        states,
        masks,
        proprio if module.proprio_dim else None,
        visual,
        valid,
    )
    state_loss = masked_state_loss(
        output.passive_next, targets, masks & valid[..., None]
    )
    loss = state_loss
    proprio_loss = loss.new_zeros(())
    visual_loss = loss.new_zeros(())
    if module.proprio_dim:
        proprio_loss = masked_state_loss(
            output.proprio_next,
            target_proprio,
            valid[..., None].expand_as(target_proprio),
        )
        loss = loss + float(config.get("proprio_loss_weight", 1.0)) * proprio_loss
    if method.visual_encoder is not None:
        with torch.no_grad():
            visual_target = method.visual_encoder(target_rgb)
        visual_loss = masked_state_loss(
            output.visual_next,
            visual_target,
            valid[..., None].expand_as(visual_target),
        )
        loss = loss + float(config.get("visual_loss_weight", 1.0)) * visual_loss
    return loss, state_loss, proprio_loss, visual_loss


@torch.inference_mode()
def _evaluate_state_prior(method, module, loader, device, config) -> dict[str, float]:
    modules = [module]
    if method.visual_encoder is not None:
        modules.append(method.visual_encoder)
    training_modes = [trainable.training for trainable in modules]
    for trainable in modules:
        trainable.eval()
    totals = np.zeros(4, np.float64)
    try:
        for batch in loader:
            losses = _state_prior_batch_losses(
                method, module, batch, device, config
            )
            totals += np.asarray([float(loss.item()) for loss in losses])
    finally:
        for trainable, was_training in zip(modules, training_modes):
            trainable.train(was_training)
    averages = totals / max(1, len(loader))
    if not np.isfinite(averages).all():
        raise FloatingPointError(f"Non-finite State Prior validation loss: {averages}")
    return dict(
        state_prior=float(averages[0]),
        state=float(averages[1]),
        proprio=float(averages[2]),
        visual=float(averages[3]),
    )


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=("state", "action"))
    parser.add_argument("--data")
    parser.add_argument("--output")
    args, unknown = parser.parse_known_args()
    overrides = _translate(unknown)
    prior = args.prior or next((item.split("=", 1)[1] for item in overrides if item.startswith("prior=")), None)
    if prior not in {"state", "action"}:
        raise ValueError("Specify prior=state or prior=action")
    overrides = [item for item in overrides if not item.startswith("prior=")]
    config = config_from_unknown(["model=prior_adapter", *overrides])
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    env_name = config["env"]["name"]
    data = Path(args.data or config.get("data", data_path(config, env_name, f"{prior}_prior")))
    dataset = LazyEpisodeDataset(
        data,
        split="train",
        seed=seed,
        max_transitions=config.get("max_transitions"),
        min_sequence_length=2 if prior == "state" else 1,
    )
    if not len(dataset):
        raise ValueError(f"No usable {prior} training episodes in {data}")
    print(f"Using {len(dataset)} train episodes ({dataset.transition_count} transitions)")
    if dataset.filtered_episode_count:
        print(
            f"Skipped {dataset.filtered_episode_count} episodes without a state transition"
        )
    sample = dataset[0]
    if prior == "state":
        config["env"]["proprio_size"] = int(sample["proprio"].shape[-1])
    method = build_method(config, sample.get("action_low"), sample.get("action_high"))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    method.to(device)
    started = time.perf_counter()
    epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", config.get("profile", {}).get("batch_size", 8)))
    if prior == "action":
        module = method.action_prior
        for parameter in module.parameters():
            parameter.requires_grad = True
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_pad_action,
        )
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(
                config["model"]["action_prior"].get(
                    "learning_rate", config.get("learning_rate", 2e-4)
                )
            ),
        )
        training_history = {"action_prior": []}
        for epoch in range(epochs):
            total = 0.0
            for actions, valid in loader:
                actions, valid = actions.to(device), valid.to(device)
                action_history = actions[:, :-1]
                output = module(action_history, valid[:, :-1])
                loss = action_prior_loss(output.distribution, actions, valid)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            average = total / max(1, len(loader))
            training_history["action_prior"].append(average)
            print(f"epoch={epoch + 1} action_prior_loss={average:.6f}")
        state = module.state_dict()
        training_summary = {
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "early_stopped": False,
        }
    else:
        module = method.state_prior
        modules = [module]
        if method.visual_encoder is not None:
            modules.append(method.visual_encoder)
        for trainable in modules:
            trainable.train()
            for parameter in trainable.parameters():
                parameter.requires_grad = True
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_pad_state,
        )
        validation_dataset = LazyEpisodeDataset(
            data,
            split="val",
            seed=seed,
            max_transitions=config.get("max_transitions"),
            min_sequence_length=2,
        )
        validation_loader = (
            DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_pad_state,
            )
            if len(validation_dataset)
            else None
        )
        monitor_name = "validation" if validation_loader is not None else "training"
        print(
            f"Monitoring {monitor_name} loss for scheduler/early stopping"
            + (
                f" ({len(validation_dataset)} validation episodes)"
                if validation_loader is not None
                else " (validation split is empty)"
            )
        )
        parameters = [
            parameter for trainable in modules for parameter in trainable.parameters()
        ]
        state_prior_config = config["model"]["state_prior"]
        learning_rate = float(
            state_prior_config.get(
                "learning_rate", config.get("learning_rate", 2e-4)
            )
        )
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        scheduler_config = dict(state_prior_config.get("scheduler", {}))
        scheduler_name = str(scheduler_config.get("name", "none")).lower()
        if scheduler_name in {"none", "off", "false"}:
            scheduler = None
        elif scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 8)),
                threshold=float(scheduler_config.get("threshold", 1e-3)),
                threshold_mode="abs",
                cooldown=int(scheduler_config.get("cooldown", 0)),
                min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            )
        else:
            raise ValueError(
                "model.state_prior.scheduler.name must be none or "
                "reduce_on_plateau"
            )
        early_config = dict(state_prior_config.get("early_stopping", {}))
        early_stopper = (
            _EarlyStopper(
                patience=int(early_config.get("patience", 30)),
                min_delta=float(early_config.get("min_delta", 1e-3)),
                warmup_epochs=int(early_config.get("warmup_epochs", 20)),
            )
            if bool(early_config.get("enabled", False))
            else None
        )
        restore_best = bool(early_config.get("restore_best", True))
        training_history = {
            "state_prior": [],
            "state": [],
            "proprio": [],
            "visual": [],
            "monitor_loss": [],
            "learning_rate": [],
        }
        if validation_loader is not None:
            training_history.update(
                {
                    "val_state_prior": [],
                    "val_state": [],
                    "val_proprio": [],
                    "val_visual": [],
                }
            )
        gradient_clip_norm = float(
            state_prior_config.get(
                "gradient_clip_norm", config.get("gradient_clip_norm", 0.0)
            )
        )
        best_state = None
        best_monitor_loss = float("inf")
        best_epoch = 0
        early_stopped = False
        epochs_completed = 0
        for epoch in range(epochs):
            total = total_state = total_proprio = total_visual = 0.0
            for batch in loader:
                loss, state_loss, proprio_loss, visual_loss = (
                    _state_prior_batch_losses(
                        method, module, batch, device, config
                    )
                )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite State Prior loss: "
                        f"state={state_loss.item()}, proprio={proprio_loss.item()}, "
                        f"visual={visual_loss.item()}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters, gradient_clip_norm, error_if_nonfinite=True
                    )
                optimizer.step()
                total += float(loss.item())
                total_state += float(state_loss.item())
                total_proprio += float(proprio_loss.item())
                total_visual += float(visual_loss.item())
            batches = max(1, len(loader))
            training_history["state_prior"].append(total / batches)
            training_history["state"].append(total_state / batches)
            training_history["proprio"].append(total_proprio / batches)
            training_history["visual"].append(total_visual / batches)
            current_lr = float(optimizer.param_groups[0]["lr"])
            training_history["learning_rate"].append(current_lr)
            if validation_loader is not None:
                validation = _evaluate_state_prior(
                    method, module, validation_loader, device, config
                )
                monitor_loss = validation["state_prior"]
                for name, value in validation.items():
                    training_history[f"val_{name}"].append(value)
            else:
                monitor_loss = total / batches
            training_history["monitor_loss"].append(monitor_loss)
            epoch_number = epoch + 1
            if early_stopper is not None:
                improved, should_stop = early_stopper.step(
                    monitor_loss, epoch_number
                )
            else:
                improved = monitor_loss < best_monitor_loss
                should_stop = False
            if improved:
                best_monitor_loss = monitor_loss
                best_epoch = epoch_number
                best_state = {
                    "state_prior": _cpu_state_dict(module),
                    "visual_encoder": _cpu_state_dict(method.visual_encoder)
                    if method.visual_encoder is not None
                    else None,
                }
            if scheduler is not None:
                scheduler.step(monitor_loss)
            next_lr = float(optimizer.param_groups[0]["lr"])
            epochs_completed = epoch_number
            print(
                f"epoch={epoch_number} state_prior_loss={total / batches:.6f} "
                f"state={total_state / batches:.6f} "
                f"proprio={total_proprio / batches:.6f} "
                f"visual={total_visual / batches:.6f} "
                f"monitor={monitor_loss:.6f} lr={current_lr:.3e}"
                + (f" next_lr={next_lr:.3e}" if next_lr != current_lr else "")
            )
            if should_stop:
                early_stopped = True
                print(
                    f"Early stopping at epoch {epoch_number}: {monitor_name} "
                    f"loss did not improve by {early_stopper.min_delta:g} for "
                    f"{early_stopper.stale_epochs} epochs; best epoch="
                    f"{early_stopper.best_epoch}, best loss="
                    f"{early_stopper.best_loss:.6f}"
                )
                break
        if restore_best and best_state is not None:
            module.load_state_dict(best_state["state_prior"])
            if method.visual_encoder is not None:
                method.visual_encoder.load_state_dict(best_state["visual_encoder"])
            print(
                f"Restored best State Prior weights from epoch {best_epoch} "
                f"({monitor_name} loss={best_monitor_loss:.6f})"
            )
        state = {
            "state_prior": module.state_dict(),
            "visual_encoder": method.visual_encoder.state_dict()
            if method.visual_encoder is not None
            else None,
        }
        training_summary = {
            "epochs_requested": epochs,
            "epochs_completed": epochs_completed,
            "early_stopped": early_stopped,
            "monitor": monitor_name,
            "best_epoch": best_epoch,
            "best_monitor_loss": best_monitor_loss,
            "initial_learning_rate": learning_rate,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "restored_best_weights": restore_best and best_state is not None,
        }
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
            "config": config,
            "state": state,
            "wall_clock_train_seconds": time.perf_counter() - started,
            "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "history": training_history,
            "training_summary": training_summary,
        },
        output,
    )
    save_training_history(
        training_history,
        output.parent / f"{output.stem}_diagnostics" / "training_loss",
    )
    print(f"Saved {prior} prior to {output}")


if __name__ == "__main__":
    main()
