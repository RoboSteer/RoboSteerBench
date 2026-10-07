"""Stable JSON/NPZ output for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metric_registry import MetricOutput, MetricSpec


def save_metric(output_root: Path, spec: MetricSpec, result: MetricOutput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": spec.name,
        "value": float(result.value),
        "direction": spec.direction,
        "num_samples": int(result.num_samples),
        **result.details,
    }
    if result.sample_values is not None:
        artifact = output_root / "metrics" / f"{_filename(spec.name)}_samples.npz"
        np.savez_compressed(
            artifact,
            sample_ids=np.asarray(result.sample_ids, dtype=np.str_),
            values=np.asarray(result.sample_values, dtype=np.float32),
        )
        payload["sample_values"] = str(artifact)
    path = output_root / "metrics" / f"{_filename(spec.name)}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["path"] = str(path)
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _filename(name: str) -> str:
    return name.lower().replace("@", "_at_").replace("-", "_").replace(" ", "_")
