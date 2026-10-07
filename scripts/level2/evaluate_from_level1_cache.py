#!/usr/bin/env python3
"""Evaluate ordinary Level-2 constraints using a saved full Level-1 text cache.

The Level-1 prediction tree may be archived. The saved phase embeddings and
per-source MM distances are subset by the same source IDs as evaluate.py;
IR_2 is still calculated from the original Level-2 rollout and shared GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.core_metric.BehaviorGenerationScore import compute_behavior_generation_score
from scripts.evaluation.core_metric.FID import motion_fid
from scripts.evaluation.data.motion import MotionIndex
from scripts.evaluation.shared.task_assets import index_task_records, match_level2_prediction_clips
from scripts.level2.ir import compute_ir2


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_cache(path: Path, value_key: str) -> dict[str, np.ndarray | float]:
    with np.load(path) as data:
        ids = data["sample_ids"].tolist()
        values = data[value_key]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return dict(zip(ids, values, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--level2-task-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--level1-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/evaluation/default.json")
    args = parser.parse_args()

    records = index_task_records(args.level2_task_root, args.dataset_root)
    matched = match_level2_prediction_clips(args.prediction, records)
    family = {record.task_family for record, _ in matched.values()}
    if len(family) != 1:
        raise ValueError(f"expected one task family, got {family}")
    family = family.pop()
    if "".join(character for character in family.casefold() if character.isalnum()) not in {
        "speed", "amplitude", "direction", "trajectory", "bodyrestrain"
    }:
        raise ValueError(f"unsupported cached family: {family}")
    print(f"matched {family}: {len(matched)} predictions", flush=True)

    cache = args.level1_cache.expanduser().resolve()
    pred_emb = _load_cache(cache / "embeddings/phase_prediction_motion.npz", "embeddings")
    gt_emb = _load_cache(cache / "embeddings/phase_motion_groundtruth_motion.npz", "embeddings")
    mm_samples = _load_cache(cache / "metrics/mm_distance_samples.npz", "values")
    linked = []
    missing = []
    for task_id, (record, prediction) in sorted(matched.items()):
        sample_id = record.sample_id
        if sample_id not in pred_emb or sample_id not in gt_emb or sample_id not in mm_samples:
            missing.append({"level2_task_id": task_id, "base_sample_id": sample_id,
                            "missing": "level1_cached_embedding_or_distance"})
            continue
        linked.append((task_id, record, prediction))
    if missing:
        raise ValueError(f"Level-1 cache missing {len(missing)} matching samples; first: {missing[:3]}")
    if not linked:
        raise RuntimeError("no linked samples")

    unique_ids = sorted({record.sample_id for _, record, _ in linked})
    fid = motion_fid(
        np.stack([gt_emb[s] for s in unique_ids]),
        np.stack([pred_emb[s] for s in unique_ids]),
    )
    mm = float(np.asarray([mm_samples[s] for s in unique_ids], dtype=np.float64).mean())
    bg = compute_behavior_generation_score(fid, mm)
    print(f"base {family}: sources={len(unique_ids)} fid={fid:.9f} mm={mm:.9f} bg={bg:.9f}", flush=True)

    predictions = {task_id: prediction for task_id, _, prediction in linked}
    groundtruth = {task_id: record.motion_groundtruth for task_id, record, _ in linked}
    metadata = {
        task_id: {
            "task_id": task_id, "task_family": family, "task_type": record.task_type,
            "task_json": str(record.task_json), "base_sample_id": record.sample_id,
            "base_motion_groundtruth": str(record.motion_groundtruth),
        }
        for task_id, record, _ in linked
    }
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    ir2 = compute_ir2(
        family=family,
        prediction=MotionIndex(args.prediction.expanduser().resolve(), predictions, {}),
        base_groundtruth=MotionIndex(args.dataset_root.expanduser().resolve() / "Data/Shared/Motion", groundtruth, {}),
        task_metadata_by_id=metadata,
        prediction_fps=float(config["prediction_fps"]),
        groundtruth_fps=float(config["motion_groundtruth_fps"]),
        device=args.device,
    )
    if set(ir2.sample_ids) != set(predictions):
        raise ValueError("IR_2 sample IDs disagree with linked predictions")
    bs = bg * ir2.value
    output = args.output.expanduser().resolve()
    provenance = {
        "source": str(cache),
        "method": "subset_saved_full_level1_phase_embeddings_and_mm_distance_by_base_sample_id",
        "level1_prediction_archived": True,
        "num_level1_cache_samples": len(pred_emb),
        "num_unique_selected_sources": len(unique_ids),
    }
    metric_dir = output / "base_generation/metrics"
    _write_json(metric_dir / "fid.json", {"name": "FID", "value": fid, "num_samples": len(unique_ids),
                "embedding_dim": 512, "unit": "complete_source_motion", "cache_provenance": provenance})
    _write_json(metric_dir / "mm_distance.json", {"name": "MM-Distance", "value": mm,
                "num_samples": len(unique_ids), "unit": "complete_source_pair", "cache_provenance": provenance})
    _write_json(metric_dir / "bg.json", {"name": "BG", "value": bg,
                "formula": "1000 * exp(-0.3 * FID - 1.6 * MM-Distance)",
                "fid": fid, "mm_distance": mm, "num_samples": len(unique_ids), "cache_provenance": provenance})
    _write_json(output / "base_generation/summary.json", {"schema": "robosteer.base_generation.cached.v1",
                "metrics": {"FID": fid, "MM-Distance": mm, "BG": bg}, "cache_provenance": provenance})
    _write_json(output / "metrics/ir_2.json", {"name": "IR_2", "value": ir2.value,
                "direction": "higher_is_better", "task_family": family, **ir2.details})
    _write_jsonl(output / "metrics/ir_2_samples.jsonl", [
        {"level2_task_id": task_id, "value": float(value), "task_family": family}
        for task_id, value in zip(ir2.sample_ids, ir2.sample_values, strict=True)
    ])
    _write_json(output / "metrics/bs_level2.json", {"name": "BS_level2", "value": bs,
                "direction": "higher_is_better", "formula": "BG(base Level-1 subset) × IR_2(Level-2)",
                "bg": bg, "ir_1": 1.0, "ir_2": ir2.value, "task_family": family})
    _write_jsonl(output / "assets/level2_assets.jsonl", [
        {"level2_task_id": task_id, "base_sample_id": record.sample_id,
         "task_family": family, "task_type": record.task_type,
         "level2_prediction": str(prediction), "level2_task_json": str(record.task_json),
         "base_motion_groundtruth": str(record.motion_groundtruth)}
        for task_id, record, prediction in linked
    ])
    result = {"schema": "robosteer.level2.v2", "task_family": family, "mm_encoder": "text_motion",
              "prediction": str(args.prediction.expanduser().resolve()),
              "base_prediction": "archived; cached Level-1 embeddings and MM distances used",
              "base_motion_groundtruth": str(args.dataset_root.expanduser().resolve() / "Data/Shared/Motion"),
              "num_level2_predictions": len(matched), "num_linked_samples": len(linked),
              "missing_base_assets": missing, "conventional_base_generation": provenance,
              "bg": bg, "ir_1": 1.0, "ir_2": ir2.value, "bs_level2": bs}
    _write_json(output / "level2_summary.json", result)
    print(json.dumps({"family": family, "bg": bg, "ir_2": ir2.value, "bs_level2": bs,
                      "samples": len(linked)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
