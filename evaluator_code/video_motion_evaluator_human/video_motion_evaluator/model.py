from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoMotionAlignment(nn.Module):
    def __init__(self, video_dim: int = 512, embedding_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(video_dim, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.normalization(self.projection(video_features)), dim=-1)
