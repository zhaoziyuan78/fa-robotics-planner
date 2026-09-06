"""Discrete frame tokenizer used by the video State Prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQVAEOutput:
    reconstruction: torch.Tensor
    tokens: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    """Straight-through nearest-neighbour vector quantizer."""

    def __init__(self, codebook_size: int, code_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.embedding = nn.Embedding(self.codebook_size, self.code_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size,
        )

    def lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return BCHW code vectors for integer BHW tokens."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,H,W]")
        return self.embedding(tokens.long()).permute(0, 3, 1, 2).contiguous()

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 4 or latent.size(1) != self.code_dim:
            raise ValueError(
                f"latent must have shape [B,{self.code_dim},H,W]"
            )
        batch, channels, height, width = latent.shape
        flat = latent.permute(0, 2, 3, 1).reshape(-1, channels)
        # ||z-e||^2 without materialising [num_latents, num_codes, code_dim].
        distances = (
            flat.square().sum(1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.square().sum(1)[None]
        )
        tokens = distances.argmin(1)
        quantized = self.embedding(tokens).view(batch, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(quantized, latent.detach())
        commitment_loss = F.mse_loss(latent, quantized.detach())
        straight_through = latent + (quantized - latent).detach()
        probabilities = F.one_hot(tokens, self.codebook_size).float().mean(0)
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities.clamp_min(1e-10))).sum()
        )
        return (
            straight_through,
            tokens.view(batch, height, width),
            codebook_loss,
            commitment_loss,
            perplexity,
        )


class VQVAE(nn.Module):
    """Three-stage x8 VQ-VAE matching the original FA-Planner tokenizer."""

    downsample_factor = 8

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        code_dim: int = 128,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        code_dim = int(code_dim)
        self.in_channels = int(in_channels)
        self.hidden_dim = hidden_dim
        self.codebook_size = int(codebook_size)
        self.code_dim = code_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, code_dim, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(
            self.codebook_size, code_dim, commitment_weight
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, self.in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def images_to_tensor(
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...] | torch.Size]:
        """Convert uint8/float HWC or CHW images with arbitrary leading dims."""

        tensor = images
        if tensor.ndim < 4:
            raise ValueError("images must include batch and HWC/CHW dimensions")
        if tensor.shape[-1] == 3:
            leading = tensor.shape[:-3]
            tensor = tensor.reshape(-1, *tensor.shape[-3:]).permute(0, 3, 1, 2)
        elif tensor.shape[-3] == 3:
            leading = tensor.shape[:-3]
            tensor = tensor.reshape(-1, *tensor.shape[-3:])
        else:
            raise ValueError("images must have three RGB channels")
        tensor = tensor.float()
        if images.dtype == torch.uint8 or float(tensor.detach().amax()) > 1.5:
            tensor = tensor / 255.0
        return tensor, leading

    @staticmethod
    def _validate_size(images: torch.Tensor) -> None:
        height, width = images.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                f"VQ-VAE image size {(height, width)} must be divisible by 8"
            )

    def forward(self, images: torch.Tensor) -> VQVAEOutput:
        images, _ = self.images_to_tensor(images)
        self._validate_size(images)
        latent = self.encoder(images)
        quantized, tokens, codebook, commitment, perplexity = self.quantizer(latent)
        reconstruction = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(reconstruction, images)
        loss = (
            reconstruction_loss
            + codebook
            + self.quantizer.commitment_weight * commitment
        )
        return VQVAEOutput(
            reconstruction,
            tokens,
            loss,
            reconstruction_loss,
            codebook,
            commitment,
            perplexity,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images, leading = self.images_to_tensor(images)
        self._validate_size(images)
        latent = self.encoder(images)
        _, tokens, _, _, _ = self.quantizer(latent)
        return tokens.reshape(*leading, *tokens.shape[-2:])

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 3:
            raise ValueError("tokens must end in [H,W]")
        leading = tokens.shape[:-2]
        flat = tokens.reshape(-1, *tokens.shape[-2:])
        reconstruction = self.decoder(self.quantizer.lookup(flat))
        return reconstruction.reshape(*leading, *reconstruction.shape[-3:])
