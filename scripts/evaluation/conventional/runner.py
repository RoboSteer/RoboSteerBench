"""Stable wrapper around the conventional metric pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def run_conventional(
    *,
    prediction: Path,
    motion_groundtruth: Path | None,
    instruction_groundtruth: Path | None,
    task_metadata: Path,
    output: Path,
    mm_encoder: str,
    metrics: Iterable[str],
    models: Path,
    config_path: Path,
    device: str = "auto",
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run conventional metrics from explicit prediction/GT/condition inputs."""
    from scripts.evaluation.pipeline.evaluator import EvaluationRequest, run_evaluation

    metrics = tuple(dict.fromkeys(metrics))
    if not metrics:
        raise ValueError("at least one conventional metric is required")
    if "BG" in metrics and ("FID" not in metrics or "MM-Distance" not in metrics):
        metrics = tuple(dict.fromkeys((*metrics, "FID", "MM-Distance")))
    config = json.loads(config_path.expanduser().resolve().read_text(encoding="utf-8"))
    config["mm_encoder"] = mm_encoder
    if config_overrides:
        config.update(config_overrides)
    request = EvaluationRequest(
        prediction=prediction.expanduser().resolve(),
        motion_groundtruth=None if motion_groundtruth is None else motion_groundtruth.expanduser().resolve(),
        instruction_groundtruth=None if instruction_groundtruth is None else instruction_groundtruth.expanduser().resolve(),
        output=output.expanduser().resolve(),
        metrics=metrics,
        models=models.expanduser().resolve(),
        task_metadata=task_metadata.expanduser().resolve(),
        config_path=config_path.expanduser().resolve(),
        device=device,
    )
    return run_evaluation(request, config)


def read_metric_value(output: Path, metric_name: str) -> float:
    """Read one metric artifact emitted by :func:`run_conventional`."""
    filename = metric_name.lower().replace("@", "_at_").replace("-", "_").replace(" ", "_")
    path = output / "metrics" / f"{filename}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != metric_name:
        raise ValueError(f"{path}: expected {metric_name!r}, got {payload.get('name')!r}")
    return float(payload["value"])
