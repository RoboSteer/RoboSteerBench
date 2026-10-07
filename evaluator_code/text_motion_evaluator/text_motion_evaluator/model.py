from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextMotionAlignment(nn.Module):
    def __init__(self, text_dim: int = 768, embedding_dim: int = 512, initial_logit_scale: float = 2.6592600369):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, embedding_dim)
        self.text_normalization = nn.LayerNorm(embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(initial_logit_scale), dtype=torch.float32))

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_normalization(self.text_projection(text_features)), dim=-1)
