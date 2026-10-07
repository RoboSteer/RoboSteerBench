from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 38, output_dim: int = 256):
        super().__init__()
        # Reuse OMG's tested temporal encoder implementation with a smaller trainable branch.
        from omg.benchmarks.evaluator.motion_encoder import MotionEncoder

        self.encoder = MotionEncoder(
            input_dim=input_dim,
            movement_dim=128,
            hidden_dim=128,
            output_dim=output_dim,
            movement_mode="linear",
            temporal_kind="transformer_rope",
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1,
            max_len=60,
        )

    def forward(
        self,
        windows: torch.Tensor,
        frame_masks: torch.Tensor,
        window_masks: torch.Tensor,
    ) -> torch.Tensor:
        if windows.ndim != 4 or frame_masks.shape != windows.shape[:3]:
            raise ValueError("Expected windows (B,W,T,D) and frame_masks (B,W,T)")
        batch_size, num_windows, frames, dimensions = windows.shape
        valid_windows = window_masks.reshape(-1).bool()
        flat_windows = windows.reshape(batch_size * num_windows, frames, dimensions)[valid_windows]
        flat_masks = frame_masks.reshape(batch_size * num_windows, frames)[valid_windows]
        encoded = self.encoder(flat_windows, valid_mask=flat_masks)
        owners = torch.arange(batch_size, device=windows.device)[:, None].expand(batch_size, num_windows).reshape(-1)
        owners = owners[valid_windows]
        pooled = torch.zeros(batch_size, encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
        pooled.index_add_(0, owners, encoded)
        counts = torch.zeros(batch_size, device=encoded.device, dtype=encoded.dtype)
        counts.index_add_(0, owners, torch.ones_like(owners, dtype=encoded.dtype))
        return F.normalize(pooled / counts.clamp_min(1.0).unsqueeze(-1), dim=-1)


class RotationMotionAlignment(nn.Module):
    def __init__(self, rotation_dim: int = 38, rotation_embedding_dim: int = 256, embedding_dim: int = 512):
        super().__init__()
        self.rotation_encoder = RotationTemporalEncoder(rotation_dim, rotation_embedding_dim)
        self.adapter_norm = nn.LayerNorm(embedding_dim + rotation_embedding_dim)
        self.adapter = nn.Linear(embedding_dim + rotation_embedding_dim, embedding_dim)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

    def encode(
        self,
        omg_embedding: torch.Tensor,
        rotation_windows: torch.Tensor,
        frame_masks: torch.Tensor,
        window_masks: torch.Tensor,
    ) -> torch.Tensor:
        rotation = self.rotation_encoder(rotation_windows, frame_masks, window_masks)
        delta = self.adapter(self.adapter_norm(torch.cat([omg_embedding, rotation], dim=-1)))
        return F.normalize(omg_embedding + delta, dim=-1)

    def forward(self, condition: dict[str, torch.Tensor], motion: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        condition_embedding = self.encode(
            condition["omg_embedding"],
            condition["rotation_windows"],
            condition["frame_masks"],
            condition["window_masks"],
        )
        motion_embedding = self.encode(
            motion["omg_embedding"],
            motion["rotation_windows"],
            motion["frame_masks"],
            motion["window_masks"],
        )
        return condition_embedding, motion_embedding
