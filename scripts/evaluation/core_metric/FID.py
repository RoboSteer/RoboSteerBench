"""Fréchet distance between reference and generated embedding distributions."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def motion_fid(reference_embeddings: np.ndarray, generated_embeddings: np.ndarray, eps: float = 1e-6) -> float:
    """Compute FID on compatible ``(N, D)`` reference/generated embeddings."""
    reference = np.asarray(reference_embeddings, dtype=np.float64)
    generated = np.asarray(generated_embeddings, dtype=np.float64)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError(f"FID requires compatible (N, D) arrays, got {reference.shape} and {generated.shape}")
    if min(reference.shape[0], generated.shape[0]) < 2:
        raise ValueError("FID requires at least two reference and two generated samples")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("FID embeddings must be finite")
    mu_ref, mu_gen = reference.mean(axis=0), generated.mean(axis=0)
    cov_ref, cov_gen = np.cov(reference, rowvar=False), np.cov(generated, rowvar=False)
    cov_mean, _ = linalg.sqrtm(cov_ref @ cov_gen, disp=False)
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_ref.shape[0], dtype=np.float64) * float(eps)
        cov_mean = linalg.sqrtm((cov_ref + offset) @ (cov_gen + offset))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    delta = mu_ref - mu_gen
    return float(max(delta @ delta + np.trace(cov_ref) + np.trace(cov_gen) - 2.0 * np.trace(cov_mean), 0.0))
