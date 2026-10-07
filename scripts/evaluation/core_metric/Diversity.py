"""Sampled and exact all-pairs embedding Diversity estimators."""

from __future__ import annotations

import numpy as np



def full_pair_diversity_gpu(
    motion_embeddings: np.ndarray,
    *,
    device: str = "auto",
    block_size: int = 4096,
) -> float:
    """Return the exact mean L2 distance over all unordered embedding pairs.

    Distances are evaluated in GPU blocks and immediately reduced into one
    float64 accumulator. The O(N**2) distance matrix is never materialized.
    """
    import torch

    embeddings = np.asarray(motion_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            "full-pair Diversity requires embeddings with shape (N, D), N >= 2; "
            f"got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("full-pair Diversity requires finite embeddings")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    requested = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU full-pair Diversity but is unavailable")
    target = torch.device(requested)
    values = torch.as_tensor(embeddings, dtype=torch.float32, device=target)
    squared_norm = torch.sum(values * values, dim=1)
    count = int(values.shape[0])
    pair_count = count * (count - 1) // 2
    total = torch.zeros((), dtype=torch.float64, device=target)
    with torch.inference_mode():
        for left_start in range(0, count, int(block_size)):
            left = values[left_start : left_start + int(block_size)]
            left_norm = squared_norm[left_start : left_start + int(block_size)]
            for right_start in range(left_start, count, int(block_size)):
                right = values[right_start : right_start + int(block_size)]
                distance_squared = (
                    left_norm[:, None]
                    + squared_norm[right_start : right_start + int(block_size)][None, :]
                    - 2.0 * (left @ right.T)
                ).clamp_min_(0.0)
                distances = torch.sqrt(distance_squared)
                if left_start == right_start:
                    total += torch.triu(distances, diagonal=1).sum(dtype=torch.float64)
                else:
                    total += distances.sum(dtype=torch.float64)
    return float((total / pair_count).item())
