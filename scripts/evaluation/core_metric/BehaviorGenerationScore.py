"""Core computation for the post-hoc Behavior Generation (BG) score."""

from __future__ import annotations

import math


DEFAULT_SCALE = 1000.0
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 1.6


def compute_behavior_generation_score(
    fid: float,
    mm_distance: float,
    *,
    scale: float = DEFAULT_SCALE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """Compute ``BG = scale * exp(-alpha * FID - beta * MM-Distance)``.

    Both source metrics are lower-is-better. The defaults were calibrated on
    the available non-audio Text, Rhythm, and Trajectory results. Their
    task-level variation contributes approximately 56.8% from FID and 43.2%
    from MM-Distance, while the scale keeps reported scores around 100.
    """
    values = {
        "fid": fid,
        "mm_distance": mm_distance,
        "scale": scale,
        "alpha": alpha,
        "beta": beta,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")

    return float(scale) * math.exp(
        -float(alpha) * float(fid) - float(beta) * float(mm_distance)
    )
