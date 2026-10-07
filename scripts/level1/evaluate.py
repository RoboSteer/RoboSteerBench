#!/usr/bin/env python3
"""Evaluate one Level-1 task from explicit prediction, GT and condition paths.

The conventional metrics always receive the paths supplied on the command line.
Task JSON is used only for task-specific IR_1 metadata, condition manifest
materialization when the condition path is a task-JSON directory, and timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
for import_root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation.conventional.runner import read_metric_value, run_conventional
from scripts.evaluation.model_paths import default_registry
from scripts.evaluation.data.motion import index_motion_root
from scripts.level1.catalog import infer_task_type
from scripts.level1.ir import compute_ir1
from scripts.evaluation.shared.task_assets import build_instruction_manifest, build_timing_manifest, index_task_records, resolve_condition


_LEVEL_METRIC_ALIASES = {"ir1", "bs1", "bslevel1"}


def evaluate_level1(args: argparse.Namespace) -> dict[str, Any]:
    prediction = args.prediction.expanduser().resolve()
    motion_groundtruth = args.motion_groundtruth.expanduser().resolve()
    condition_groundtruth = args.condition_groundtruth.expanduser().resolve()
    output = args.output.expanduser().resolve()
    selected_ids = set(index_motion_root(prediction).samples) & set(index_motion_root(motion_groundtruth).samples)
    if not selected_ids:
        raise RuntimeError("prediction and motion ground truth have no matching sample IDs")

    task_type = args.task_type
    task_records = None
    if args.task_root is not None:
        dataset_root = _dataset_root(args)
        task_records = index_task_records(args.task_root, dataset_root)
        if task_type is None:
            task_type = infer_task_type(args.task_root)
        task_metadata = build_timing_manifest(
            [record for record in task_records if record.sample_id in selected_ids],
            output / "assets" / "timing.jsonl",
        )
    elif args.task_metadata is not None:
        task_metadata = args.task_metadata.expanduser().resolve()
    else:
        raise ValueError("provide --task-root or --task-metadata for reproducible timing/IR_1")
    if task_type is None:
        raise ValueError("provide --task-type when --task-root cannot infer one")

    instruction_groundtruth, condition_source = _prepare_condition_input(
        condition_groundtruth=condition_groundtruth,
        modality=args.modality,
        dataset_root=args.dataset_root,
        selected_ids=selected_ids,
        destination=output / "assets" / "conditions.json",
    )
    metrics = _conventional_metrics(args.metrics)
    summary = run_conventional(
        prediction=prediction,
        motion_groundtruth=motion_groundtruth,
        instruction_groundtruth=instruction_groundtruth,
        task_metadata=task_metadata,
        output=output,
        mm_encoder=args.mm_encoder,
        metrics=metrics,
        models=args.models,
        config_path=args.config,
        device=args.device,
        config_overrides=_runtime_config_overrides(args),
    )
    bg = read_metric_value(output, "BG")
    ir1, details = compute_ir1(task_type, output, keyframe_task_root=args.keyframe_task_root)
    _write_json(output / "metrics" / "ir_1.json", {
        "name": "IR_1", "value": ir1, "direction": "higher_is_better",
        "task_type": task_type, **details,
    })
    bs = bg * ir1
    _write_json(output / "metrics" / "bs_level1.json", {
        "name": "BS_level1", "value": bs, "direction": "higher_is_better",
        "formula": "BG × IR_1(task_type)", "task_type": task_type,
        "bg": bg, "ir_1": ir1,
    })
    result = {
        "schema": "robosteer.level1.v2",
        "task_type": task_type,
        "mm_encoder": args.mm_encoder,
        "prediction": str(prediction),
        "motion_groundtruth": str(motion_groundtruth),
        "condition_groundtruth": condition_source,
        "num_prediction_gt_pairs": len(selected_ids),
        "conventional": summary,
        "bg": bg, "ir_1": ir1, "bs_level1": bs,
    }
    _write_json(output / "level1_summary.json", result)
    return result


def _prepare_condition_input(*, condition_groundtruth: Path, modality: str | None,
                             dataset_root: Path | None, selected_ids: set[str], destination: Path) -> tuple[Path, str]:
    """Use a supplied manifest/raw root directly, or materialize a task-JSON condition view."""
    if condition_groundtruth.is_file() or (condition_groundtruth / "manifest.json").is_file():
        return condition_groundtruth, str(condition_groundtruth)
    json_paths = tuple(condition_groundtruth.rglob("*.json")) if condition_groundtruth.is_dir() else ()
    if not json_paths:
        return condition_groundtruth, str(condition_groundtruth)
    if modality is None:
        raise ValueError("--modality is required when --condition-groundtruth is a task-JSON directory")
    if dataset_root is None:
        raise ValueError("--dataset-root is required when --condition-groundtruth is a task-JSON directory")
    records = index_task_records(condition_groundtruth, dataset_root)
    selected = []
    for record in records:
        if record.sample_id not in selected_ids:
            continue
        try:
            resolve_condition(record, modality, dataset_root)
        except (ValueError, FileNotFoundError):
            # A shared task directory can contain several modalities for the
            # same motion ID. Only keep records carrying the requested one.
            continue
        selected.append(record)
    if not selected:
        raise RuntimeError(f"condition task JSON has no {modality!r} record matching prediction/GT")
    return build_instruction_manifest(selected, modality=modality, dataset_root=dataset_root, destination=destination), str(condition_groundtruth)


def _dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required when --task-root is supplied")
    return args.dataset_root.expanduser().resolve()


def _conventional_metrics(groups: list[str]) -> tuple[str, ...]:
    values = [name.strip() for group in groups for name in group.split(",") if name.strip()]
    metrics = tuple(name for name in values if _normal(name) not in _LEVEL_METRIC_ALIASES)
    return tuple(dict.fromkeys((*metrics, "BG")))


def _runtime_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.mm_gpus:
        gpu_ids = [item.strip() for item in args.mm_gpus.split(",") if item.strip()]
        if not gpu_ids or any(not item.isdigit() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("--mm-gpus must be unique comma-separated non-negative GPU indices")
        overrides["mm_gpu_devices"] = [f"cuda:{int(item)}" for item in gpu_ids]
    if args.video_cache_root is not None:
        overrides["video_local_cache_root"] = str(args.video_cache_root.expanduser().resolve())
    if args.mm_video_decode_workers_per_gpu is not None:
        if args.mm_video_decode_workers_per_gpu <= 0:
            raise ValueError("--mm-video-decode-workers-per-gpu must be positive")
        overrides["mm_video_decode_workers_per_gpu"] = args.mm_video_decode_workers_per_gpu
    return overrides


def _normal(name: str) -> str:
    return "".join(character.casefold() for character in name if character.isalnum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--motion-groundtruth", type=Path, required=True)
    parser.add_argument("--condition-groundtruth", type=Path, required=True,
                        help="condition manifest/raw asset root, or task-JSON directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--mm-encoder", required=True)
    parser.add_argument("--mm-gpus", help="comma-separated GPUs for MM encoding")
    parser.add_argument("--video-cache-root", type=Path)
    parser.add_argument("--mm-video-decode-workers-per-gpu", type=int)
    parser.add_argument("--task-root", type=Path, help="Level-1 task JSON root for IR_1 and timing")
    parser.add_argument("--task-type", choices=("motion_generation", "motion_fore", "motion_retro", "motion_inter", "keyframe_conditioning", "target_reaching", "upper_full", "lower_full"))
    parser.add_argument("--task-metadata", type=Path, help="timing JSONL when no task root is used")
    parser.add_argument("--dataset-root", type=Path, help="dataset root for task-JSON resolution")
    parser.add_argument("--modality", choices=("text", "audio", "rhythm", "trajectory", "human_video", "skeleton_video"))
    parser.add_argument("--models", type=Path, default=default_registry(PROJECT_ROOT))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation" / "default.json")
    parser.add_argument("--keyframe-task-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate_level1(parse_args()), ensure_ascii=False, indent=2))
