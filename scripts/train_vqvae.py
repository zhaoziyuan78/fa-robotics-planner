from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from fa_robotics_planner.data import LazyEpisodeDataset
from fa_robotics_planner.models.vqvae import VQVAE
from fa_robotics_planner.utils import seed_everything

from ._common import checkpoint_path, config_from_unknown, data_path


def _build_tokenizer(config) -> VQVAE:
    tokenizer = config["model"]["tokenizer"]
    return VQVAE(
        hidden_dim=int(tokenizer.get("hidden_dim", 128)),
        codebook_size=int(tokenizer.get("codebook_size", 512)),
        code_dim=int(tokenizer.get("code_dim", 128)),
        commitment_weight=float(tokenizer.get("commitment_weight", 0.25)),
    )


def _episode_frames(episode: dict[str, np.ndarray]) -> torch.Tensor:
    frames = [torch.from_numpy(episode["rgb"])]
    if "next_rgb" in episode:
        frames.append(torch.from_numpy(episode["next_rgb"][-1:]))
    return torch.cat(frames, dim=0)


@torch.inference_mode()
def _validate(model, dataset, device, frame_batch_size: int) -> dict[str, float]:
    model.eval()
    totals = np.zeros(5, np.float64)
    batches = 0
    for episode in dataset:
        frames = _episode_frames(episode)
        for start in range(0, len(frames), frame_batch_size):
            output = model(frames[start : start + frame_batch_size].to(device))
            totals += np.asarray(
                [
                    output.loss.item(),
                    output.reconstruction_loss.item(),
                    output.codebook_loss.item(),
                    output.commitment_loss.item(),
                    output.perplexity.item(),
                ]
            )
            batches += 1
    averages = totals / max(1, batches)
    return dict(
        loss=float(averages[0]),
        reconstruction=float(averages[1]),
        codebook=float(averages[2]),
        commitment=float(averages[3]),
        perplexity=float(averages[4]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", *unknown])
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    env_name = config["env"]["name"]
    root = Path(args.data or config.get("data", data_path(config, env_name, "state_prior")))
    train = LazyEpisodeDataset(
        root,
        split="train",
        seed=seed,
        max_transitions=config.get("max_transitions"),
    )
    validation = LazyEpisodeDataset(
        root,
        split="val",
        seed=seed,
        max_transitions=config.get("max_transitions"),
    )
    if not len(train):
        raise ValueError(f"No VQ-VAE training episodes in {root}")
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model = _build_tokenizer(config).to(device)
    tokenizer_config = config["model"]["tokenizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tokenizer_config.get("learning_rate", 2e-4))
    )
    epochs = int(config.get("epochs", tokenizer_config.get("epochs", 20)))
    frame_batch_size = int(config.get("batch_size", 64))
    gradient_clip = float(tokenizer_config.get("gradient_clip_norm", 1.0))
    history = {"loss": [], "reconstruction": [], "codebook": [], "commitment": [], "perplexity": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None
    started = time.perf_counter()
    example = None
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        totals = np.zeros(5, np.float64)
        batches = 0
        progress = tqdm(order, desc=f"vqvae {epoch + 1}/{epochs}", unit="episode")
        for index in progress:
            frames = _episode_frames(train[int(index)])
            permutation = torch.randperm(len(frames))
            for start in range(0, len(frames), frame_batch_size):
                batch = frames[permutation[start : start + frame_batch_size]].to(device)
                output = model(batch)
                optimizer.zero_grad(set_to_none=True)
                output.loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                totals += np.asarray(
                    [output.loss.item(), output.reconstruction_loss.item(), output.codebook_loss.item(), output.commitment_loss.item(), output.perplexity.item()]
                )
                batches += 1
                original_chw, _ = model.images_to_tensor(batch[:8])
                example = (
                    original_chw.detach(),
                    output.reconstruction[:8].detach(),
                )
            progress.set_postfix(loss=f"{totals[0] / max(1, batches):.4f}", refresh=False)
        averages = totals / max(1, batches)
        validation_metrics = _validate(
            model, validation if len(validation) else train, device, frame_batch_size
        )
        for name, value in zip(("loss", "reconstruction", "codebook", "commitment", "perplexity"), averages):
            history[name].append(float(value))
        history["val_loss"].append(validation_metrics["loss"])
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"epoch={epoch + 1} loss={averages[0]:.6f} recon={averages[1]:.6f} "
            f"codebook={averages[2]:.6f} commitment={averages[3]:.6f} "
            f"perplexity={averages[4]:.2f} val={validation_metrics['loss']:.6f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    output_path = Path(
        args.output
        or config.get(
            "output",
            checkpoint_path(config, "tokenizers", f"{env_name}_vqvae_seed{seed}.pt"),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "vqvae",
            "architecture_version": 1,
            "config": config,
            "tokenizer": model.state_dict(),
            "history": history,
            "best_validation_loss": best_loss,
            "wall_clock_train_seconds": time.perf_counter() - started,
        },
        output_path,
    )
    if example is not None:
        original, reconstruction = example
        panel = torch.cat((original, reconstruction), dim=-1)
        panel = (panel.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        iio.imwrite(output_path.with_name(output_path.stem + "_reconstruction.png"), np.concatenate(panel, axis=0))
    print(f"Saved VQ-VAE to {output_path}")


if __name__ == "__main__":
    main()
