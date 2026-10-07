from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmMotionAlignment(nn.Module):
    def __init__(self, audio_dim: int, embedding_dim: int = 512, initial_logit_scale: float = 2.6592600369):
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, embedding_dim)
        self.audio_normalization = nn.LayerNorm(embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(initial_logit_scale), dtype=torch.float32))

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.audio_normalization(self.audio_projection(audio_features)), dim=-1)
