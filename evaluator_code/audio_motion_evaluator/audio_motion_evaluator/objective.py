from __future__ import annotations

import torch
import torch.nn as nn


def semantic_equality_mask(semantic_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in semantic_ids] for left in semantic_ids],
        dtype=torch.bool,
        device=device,
    )


def _masked_pair_cross_entropy(logits: torch.Tensor, semantic_equal: torch.Tensor) -> torch.Tensor:
    """Diagonal pair is positive; same-caption non-pairs are ignored."""
    if logits.shape != semantic_equal.shape or logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Logits and semantic mask must be equal square matrices")
    diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    valid_candidates = ~semantic_equal | diagonal
    denominator = torch.logsumexp(logits.masked_fill(~valid_candidates, -torch.inf), dim=1)
    positive = torch.diagonal(logits)
    return (denominator - positive).mean()


def symmetric_masked_pair_infonce(
    audio_embeddings: torch.Tensor,
    motion_embeddings: torch.Tensor,
    semantic_ids: list[str],
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if audio_embeddings.shape != motion_embeddings.shape:
        raise ValueError("Audio and motion embedding batches must have the same shape")
    semantic_equal = semantic_equality_mask(semantic_ids, audio_embeddings.device)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * audio_embeddings @ motion_embeddings.T
    audio_to_motion = _masked_pair_cross_entropy(logits, semantic_equal)
    motion_to_audio = _masked_pair_cross_entropy(logits.T, semantic_equal.T)
    loss = 0.5 * (audio_to_motion + motion_to_audio)
    diagonal = torch.eye(len(semantic_ids), dtype=torch.bool, device=audio_embeddings.device)
    ignored = semantic_equal & ~diagonal
    return loss, {
        "loss": float(loss.detach()),
        "audio_to_motion": float(audio_to_motion.detach()),
        "motion_to_audio": float(motion_to_audio.detach()),
        "logit_scale": float(scale.detach()),
        "mean_ignored_same_semantic_negatives": float(ignored.sum(dim=1).float().mean()),
    }


class AudioMotionAlignment(nn.Module):
    def __init__(self, audio_dim: int, embedding_dim: int = 512, initial_logit_scale: float | None = None):
        super().__init__()
        from .encoders import AudioProjection

        self.audio_projection = AudioProjection(audio_dim, embedding_dim)
        initial = float(initial_logit_scale) if initial_logit_scale is not None else 2.6592600369
        self.logit_scale = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.audio_projection(audio_features)
