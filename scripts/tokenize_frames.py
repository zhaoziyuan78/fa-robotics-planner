from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fa_robotics_planner.data import EpisodeWriter, LazyEpisodeDataset
from fa_robotics_planner.data.schemas import DatasetKind

from ._common import checkpoint_path, config_from_unknown, data_path
from .train_vqvae import _build_tokenizer


@torch.inference_mode()
def _encode(model, frames: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    encoded = []
    for start in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[start : start + batch_size]).to(device)
        encoded.append(model.encode(batch).cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--vqvae")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", *unknown])
    env_name = config["env"]["name"]
    source = Path(args.data or config.get("data", data_path(config, env_name, "state_prior")))
    output = Path(args.output or data_path(config, env_name, "tokens/state_prior"))
    seed = int(config.get("seed", 0))
    checkpoint_file = Path(
        args.vqvae
        or config.get(
            "vqvae_checkpoint",
            checkpoint_path(config, "tokenizers", f"{env_name}_vqvae_seed{seed}.pt"),
        )
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "vqvae":
        raise ValueError(f"Not a VQ-VAE checkpoint: {checkpoint_file}")
    model = _build_tokenizer(config)
    model.load_state_dict(checkpoint["tokenizer"])
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()
    dataset = LazyEpisodeDataset(source)
    writer = EpisodeWriter(
        output,
        DatasetKind.TOKENS,
        {
            "source": str(source),
            "source_kind": dataset.kind,
            "vqvae": str(checkpoint_file),
            "codebook_size": model.codebook_size,
            "downsample_factor": model.downsample_factor,
        },
    )
    batch_size = int(config.get("batch_size", 128))
    for index in tqdm(range(len(dataset)), desc="tokenize", unit="episode"):
        episode = dataset[index]
        arrays = {
            "video_tokens": _encode(model, episode["rgb"], device, batch_size),
            "sequence_length": np.asarray(episode["sequence_length"], np.int64),
        }
        if "next_rgb" in episode:
            arrays["next_video_tokens"] = _encode(
                model, episode["next_rgb"], device, batch_size
            )
        writer.write(int(dataset.entries[index]["id"]), arrays)
    print(f"Wrote token cache to {output}")


if __name__ == "__main__":
    main()
