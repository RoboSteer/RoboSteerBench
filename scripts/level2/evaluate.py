#!/usr/bin/env python3
"""Evaluate one Level-2 constraint family against its Level-1 base subset.

``--prediction`` is used only for IR_2.  FID, MM-Distance and BG are computed
from matching IDs in ``--base-prediction``, ``--base-motion-groundtruth`` and
``--base-condition-groundtruth``; therefore IR_1 is exactly one for Level 2.
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
from scripts.evaluation.data.motion import MotionIndex, index_motion_root
from scripts.level1.evaluate import _prepare_condition_input, _runtime_config_overrides
from scripts.level2.ir import compute_ir2
from scripts.evaluation.shared.task_assets import build_instruction_manifest, build_timing_manifest, index_task_records, materialize_motion_root, match_level2_prediction_clips

_LEVEL_METRIC_ALIASES = {"ir1", "ir2", "bs1", "bs2", "bslevel1", "bslevel2"}


def evaluate_level2(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    level2_records = index_task_records(args.level2_task_root, dataset_root)
    matched_level2 = match_level2_prediction_clips(
        args.prediction, level2_records,
        allow_unmatched=args.modality in {"human_video", "skeleton_video"},
    )
    base_predictions = index_motion_root(args.base_prediction).samples
    base_groundtruth = index_motion_root(args.base_motion_groundtruth).samples

    linked: list[tuple[Any, Path, Path, Path]] = []
    missing: list[dict[str, str]] = []
    for task_id, (record, level2_prediction) in sorted(matched_level2.items()):
        base_prediction = base_predictions.get(record.sample_id)
        base_gt = base_groundtruth.get(record.sample_id)
        if base_prediction is None or base_gt is None:
            missing.append({"level2_task_id": task_id, "base_sample_id": record.sample_id,
                            "missing": "base_prediction" if base_prediction is None else "base_motion_groundtruth"})
            continue
        linked.append((record, level2_prediction, base_prediction, base_gt))
    if not linked:
        raise RuntimeError("no Level-2 rollout has both matching Level-1 prediction and GT")
    families = {record.task_family for record, _, _, _ in linked}
    if len(families) != 1:
        raise ValueError(f"Level-2 evaluation requires one constraint family, found {sorted(families)}")
    family = next(iter(families))

    output = args.output.expanduser().resolve()
    assets = output / "assets"
    selected_ids = {record.sample_id for record, _, _, _ in linked}
    base_prediction_root = materialize_motion_root(
        {record.sample_id: base_prediction for record, _, base_prediction, _ in linked}, assets / "base_prediction"
    )
    base_gt_root = materialize_motion_root(
        {record.sample_id: base_gt for record, _, _, base_gt in linked}, assets / "base_motion_groundtruth"
    )
    # The paired Video Level-2 tasks carry the same original source-video
    # condition as Level 1. Reuse their indexed records instead of scanning
    # the large Level-1 task tree again for every branch.
    if args.modality in {"human_video", "skeleton_video"} and family.casefold() != "trajectory":
        unique_records = {record.sample_id: record for record, _, _, _ in linked}
        base_condition = build_instruction_manifest(
            unique_records.values(), modality=args.modality, dataset_root=dataset_root,
            destination=assets / "base_conditions.json",
        )
        condition_source = str(args.level2_task_root.expanduser().resolve())
    else:
        base_condition, condition_source = _prepare_condition_input(
            condition_groundtruth=args.base_condition_groundtruth.expanduser().resolve(),
            modality=args.modality,
            dataset_root=dataset_root,
            selected_ids=selected_ids,
            destination=assets / "base_conditions.json",
        )
    base_timing = build_timing_manifest(
        {record.sample_id: record for record, _, _, _ in linked}.values(),
        assets / "base_timing.jsonl"
    )
    conventional_summary = run_conventional(
        prediction=base_prediction_root,
        motion_groundtruth=base_gt_root,
        instruction_groundtruth=base_condition,
        task_metadata=base_timing,
        output=output / "base_generation",
        mm_encoder=args.mm_encoder,
        metrics=_conventional_metrics(args.metrics),
        models=args.models,
        config_path=args.config,
        device=args.device,
        config_overrides=_runtime_config_overrides(args),
    )
    bg = read_metric_value(output / "base_generation", "BG")

    prediction_index = MotionIndex(args.prediction.expanduser().resolve(), {record.task_id: prediction for record, prediction, _, _ in linked}, {})
    groundtruth_index = MotionIndex(args.base_motion_groundtruth.expanduser().resolve(), {record.task_id: gt for record, _, _, gt in linked}, {})
    metadata = {
        record.task_id: {"task_id": record.task_id, "task_family": record.task_family,
                         "task_type": record.task_type, "task_json": str(record.task_json),
                         "base_sample_id": record.sample_id, "base_motion_groundtruth": str(gt)}
        for record, _, _, gt in linked
    }
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    ir2 = compute_ir2(family=family, prediction=prediction_index, base_groundtruth=groundtruth_index,
                      task_metadata_by_id=metadata, prediction_fps=float(config["prediction_fps"]),
                      groundtruth_fps=float(config["motion_groundtruth_fps"]), device=args.device)
    _write_jsonl(assets / "level2_assets.jsonl", linked)
    _write_json(output / "metrics" / "ir_2.json", {"name": "IR_2", "value": ir2.value,
        "direction": "higher_is_better", "task_family": family, **ir2.details})
    _write_jsonl(output / "metrics" / "ir_2_samples.jsonl", [
        {"level2_task_id": task_id, "value": float(value), "task_family": family}
        for task_id, value in zip(ir2.sample_ids, ir2.sample_values, strict=True)
    ])
    bs = bg * ir2.value
    _write_json(output / "metrics" / "bs_level2.json", {"name": "BS_level2", "value": bs,
        "direction": "higher_is_better", "formula": "BG(base Level-1 subset) × IR_2(Level-2)",
        "bg": bg, "ir_1": 1.0, "ir_2": ir2.value, "task_family": family})
    result = {"schema": "robosteer.level2.v2", "task_family": family, "mm_encoder": args.mm_encoder,
        "prediction": str(args.prediction.expanduser().resolve()), "base_prediction": str(args.base_prediction.expanduser().resolve()),
        "base_motion_groundtruth": str(args.base_motion_groundtruth.expanduser().resolve()),
        "base_condition_groundtruth": condition_source, "num_level2_predictions": len(matched_level2),
        "num_linked_samples": len(linked), "missing_base_assets": missing,
        "conventional_base_generation": conventional_summary, "bg": bg, "ir_1": 1.0,
        "ir_2": ir2.value, "bs_level2": bs}
    _write_json(output / "level2_summary.json", result)
    return result


def _conventional_metrics(groups: list[str]) -> tuple[str, ...]:
    values = [name.strip() for group in groups for name in group.split(",") if name.strip()]
    metrics = tuple(name for name in values if _normal(name) not in _LEVEL_METRIC_ALIASES)
    return tuple(dict.fromkeys((*metrics, "BG")))


def _normal(name: str) -> str:
    return "".join(character.casefold() for character in name if character.isalnum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Any] | list[tuple[Any, Path, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = []
    for row in rows:
        if isinstance(row, dict):
            serialized.append(json.dumps(row, ensure_ascii=False))
            continue
        record, prediction, base_prediction, base_gt = row
        serialized.append(json.dumps({"level2_task_id": record.task_id, "base_sample_id": record.sample_id,
            "task_family": record.task_family, "task_type": record.task_type, "level2_prediction": str(prediction),
            "level2_task_json": str(record.task_json), "base_prediction": str(base_prediction),
            "base_motion_groundtruth": str(base_gt)}, ensure_ascii=False))
    path.write_text("\n".join(serialized) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True, help="Level-2 constrained motion prediction root")
    parser.add_argument("--level2-task-root", type=Path, required=True, help="one Level-2 constraint-family/modality task JSON root")
    parser.add_argument("--base-prediction", type=Path, required=True, help="same model's Level-1 generation prediction root")
    parser.add_argument("--base-motion-groundtruth", type=Path, required=True, help="Level-1 generation GT motion root")
    parser.add_argument("--base-condition-groundtruth", type=Path, required=True, help="Level-1 condition manifest/raw root, or task-JSON root")
    parser.add_argument("--dataset-root", type=Path, required=True, help="SteerableMotionBenchmarkDataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--mm-encoder", required=True)
    parser.add_argument("--modality", choices=("text", "audio", "rhythm", "trajectory", "human_video", "skeleton_video"),
                        help="needed only when base-condition-groundtruth is a task-JSON directory")
    parser.add_argument("--mm-gpus", help="comma-separated GPUs for MM encoding")
    parser.add_argument("--video-cache-root", type=Path)
    parser.add_argument("--mm-video-decode-workers-per-gpu", type=int)
    parser.add_argument("--models", type=Path, default=default_registry(PROJECT_ROOT))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation" / "default.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate_level2(parse_args()), ensure_ascii=False, indent=2))
