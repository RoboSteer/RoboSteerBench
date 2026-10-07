"""Top-level dependency-driven evaluator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.data.motion import index_motion_root

from tqdm.auto import tqdm

from .context import EvaluationContext
from .metric_registry import (
    INSTRUCTION_GROUNDTRUTH,
    MOTION_GROUNDTRUTH,
    MetricSpec,
    resolve_metrics,
)
from .report import save_json, save_metric


@dataclass(frozen=True)
class EvaluationRequest:
    prediction: Path
    motion_groundtruth: Path | None
    instruction_groundtruth: Path | None
    output: Path
    metrics: tuple[str, ...]
    models: Path
    task_metadata: Path
    config_path: Path
    device: str = "auto"
    # Internal only: automatic dual Direction evaluation sets this to a side.
    direction_task_type: str | None = None


def run_evaluation(request: EvaluationRequest, config: dict[str, Any]) -> dict[str, Any]:
    specs = resolve_metrics(list(request.metrics))
    _validate_inputs(request, specs)
    if request.direction_task_type is None and _has_ambiguous_bare_direction_ids(request):
        output = request.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        summaries = {
            side: run_evaluation(
                replace(request, output=output / side, direction_task_type=side), config
            )
            for side in ("left", "right")
        }
        summary = {
            "schema": "liujiahui.evaluation.direction_dual_summary.v1",
            "prediction": str(request.prediction.resolve()),
            "motion_groundtruth": _path(request.motion_groundtruth),
            "metrics": [spec.name for spec in specs],
            "separate_evaluations": summaries,
        }
        save_json(output / "direction_both_summary.json", summary)
        return summary
    context = EvaluationContext(
        prediction=request.prediction,
        motion_groundtruth=request.motion_groundtruth,
        instruction_groundtruth=request.instruction_groundtruth,
        output=request.output,
        models=request.models,
        task_metadata=request.task_metadata,
        config=config,
        device=request.device,
        direction_task_type=request.direction_task_type,
    )
    run_config = {
        "schema": "liujiahui.evaluation.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction": str(request.prediction.resolve()),
        "motion_groundtruth": _path(request.motion_groundtruth),
        "instruction_groundtruth": _path(request.instruction_groundtruth),
        "output": str(request.output.resolve()),
        "metrics": [spec.name for spec in specs],
        "models": str(request.models.resolve()),
        "task_metadata": str(request.task_metadata.resolve()),
        "task_metadata_duration_field": "metadata.duration",
        "config": str(request.config_path.resolve()),
        "protocol": config,
        "device": request.device,
        "direction_task_type": request.direction_task_type,
    }
    save_json(request.output.resolve() / "run_config.json", run_config)
    metric_results: dict[str, Any] = {}
    for spec in tqdm(specs, desc="Computing metrics", unit="metric", dynamic_ncols=True):
        result = spec.run(context)
        context.metric_outputs[spec.name] = result
        metric_results[spec.name] = save_metric(request.output.resolve(), spec, result)
    summary = {
        "schema": "liujiahui.evaluation.summary.v1",
        "metrics": metric_results,
        "data": context.report(),
        "run_config": str(request.output.resolve() / "run_config.json"),
        "manifest": str(request.output.resolve() / "manifest.jsonl"),
    }
    save_json(request.output.resolve() / "summary.json", summary)
    return summary


def _validate_inputs(request: EvaluationRequest, specs: list[MetricSpec]) -> None:
    requirements = set().union(*(spec.requires for spec in specs))
    if MOTION_GROUNDTRUTH in requirements and request.motion_groundtruth is None:
        raise ValueError(
            f"metrics {', '.join(spec.name for spec in specs if MOTION_GROUNDTRUTH in spec.requires)} "
            "require --motion-groundtruth"
        )
    if INSTRUCTION_GROUNDTRUTH in requirements and request.instruction_groundtruth is None:
        raise ValueError(
            f"metrics {', '.join(spec.name for spec in specs if INSTRUCTION_GROUNDTRUTH in spec.requires)} "
            "require --instruction-groundtruth"
        )


def _path(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def _has_ambiguous_bare_direction_ids(request: EvaluationRequest) -> bool:
    """Return true only for a pure bare-ID Direction export with both sides.

    The split is deliberately all-or-nothing: mixed qualified and bare IDs
    still fail the normal matcher instead of producing a silently partial run.
    """
    task_root = request.motion_groundtruth
    if task_root is None or not task_root.is_dir():
        return False
    tasks_by_source: dict[str, set[str]] = {}
    try:
        json_paths = sorted(task_root.glob("*.json"))
        for path in json_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            ground_truth = payload.get("ground_truth", {})
            family = "".join(
                character for character in str(metadata.get("task_family", "")).casefold()
                if character.isalnum()
            )
            task_type = str(metadata.get("task_type", "")).casefold()
            motion_path = ground_truth.get("motion_parameters")
            if family != "direction" or task_type not in {"left", "right"} or not isinstance(motion_path, str):
                continue
            source = _normal_id(Path(motion_path).name)
            tasks_by_source.setdefault(source, set()).add(task_type)
        prediction_ids = {_normal_id(sample_id) for sample_id in index_motion_root(request.prediction).samples}
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(prediction_ids) and all(
        tasks_by_source.get(sample_id) == {"left", "right"}
        for sample_id in prediction_ids
    )


def _normal_id(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())
