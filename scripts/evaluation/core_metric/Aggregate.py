"""Aggregation of per-sample traditional metric values."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite 1D sequence")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "min": float(array.min()), "max": float(array.max()), "num_samples": int(array.size)}
