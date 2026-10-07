"""Paired instruction--motion embedding distance."""

from __future__ import annotations

import numpy as np


def paired_mm_distance(instruction_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean paired L2 distance and the per-sample distances.

    Row ``i`` in each array must refer to the same sample ID. This function
    never performs nearest-neighbour matching or encoder loading.
    """
    instruction = np.asarray(instruction_embeddings, dtype=np.float64)
    motion = np.asarray(motion_embeddings, dtype=np.float64)
    if instruction.ndim != 2 or motion.ndim != 2:
        raise ValueError(f"MM-Distance requires (N, D) arrays, got {instruction.shape} and {motion.shape}")
    if instruction.shape != motion.shape:
        raise ValueError(f"paired embeddings require identical shape, got {instruction.shape} and {motion.shape}")
    if instruction.shape[0] == 0 or not np.isfinite(instruction).all() or not np.isfinite(motion).all():
        raise ValueError("MM-Distance requires non-empty finite embeddings")
    distances = np.linalg.norm(instruction - motion, axis=1)
    return float(distances.mean()), distances
