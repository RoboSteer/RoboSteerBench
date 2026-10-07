#!/usr/bin/env python3
"""Recompute a Level-2 IR_2 and BS_2 from a prior matched-assets manifest.

This preserves the original Level-1 BG when its source prediction has since
been archived. It uses the same compute_ir2 dispatch as evaluate.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.data.motion import MotionIndex
from scripts.level2.ir import compute_ir2


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/evaluation/default.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    summary_path = output / "level2_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["task_family"]
    if family not in {"Direction", "Trajectory"}:
        raise ValueError(f"unsupported family for this recomputation: {family}")
    if summary["missing_base_assets"]:
        raise ValueError("prior evaluation had missing base assets")

    manifest = output / "assets/level2_assets.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len(rows) != summary["num_linked_samples"] or len(rows) != summary["num_level2_predictions"]:
        raise ValueError("matched manifest count disagrees with prior summary")
    predictions: dict[str, Path] = {}
    groundtruth: dict[str, Path] = {}
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row["level2_task_id"]
        prediction = Path(row["level2_prediction"])
        gt = Path(row["base_motion_groundtruth"])
        task_json = Path(row["level2_task_json"])
        if task_id in predictions or row["task_family"] != family:
            raise ValueError(f"duplicate or wrong-family task: {task_id}")
        if not prediction.is_dir() or not gt.is_dir() or not task_json.is_file():
            raise FileNotFoundError(f"missing prediction, GT, or task JSON: {task_id}")
        predictions[task_id] = prediction
        groundtruth[task_id] = gt
        metadata[task_id] = {
            "task_id": task_id,
            "task_family": family,
            "task_type": row["task_type"],
            "task_json": str(task_json),
            "base_sample_id": row["base_sample_id"],
            "base_motion_groundtruth": str(gt),
        }

    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    ir2 = compute_ir2(
        family=family,
        prediction=MotionIndex(Path(summary["prediction"]), predictions, {}),
        base_groundtruth=MotionIndex(Path(summary["base_motion_groundtruth"]), groundtruth, {}),
        task_metadata_by_id=metadata,
        prediction_fps=float(config["prediction_fps"]),
        groundtruth_fps=float(config["motion_groundtruth_fps"]),
        device=args.device,
    )
    if len(ir2.sample_ids) != len(rows) or set(ir2.sample_ids) != set(predictions):
        raise ValueError("recomputed IR_2 sample IDs disagree with matched manifest")

    bg_path = output / "base_generation/metrics/bg.json"
    bg = float(json.loads(bg_path.read_text(encoding="utf-8"))["value"])
    if not math.isclose(bg, float(summary["bg"]), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("prior BG metric disagrees with Level-2 summary")
    bs = bg * ir2.value
    metrics = output / "metrics"
    _write_json(metrics / "ir_2.json", {
        "name": "IR_2", "value": ir2.value, "direction": "higher_is_better",
        "task_family": family, **ir2.details,
    })
    _write_jsonl(metrics / "ir_2_samples.jsonl", [
        {"level2_task_id": task_id, "value": float(value), "task_family": family}
        for task_id, value in zip(ir2.sample_ids, ir2.sample_values, strict=True)
    ])
    _write_json(metrics / "bs_level2.json", {
        "name": "BS_level2", "value": bs, "direction": "higher_is_better",
        "formula": "BG(base Level-1 subset) × IR_2(Level-2)",
        "bg": bg, "ir_1": 1.0, "ir_2": ir2.value, "task_family": family,
    })
    summary["ir_2"] = ir2.value
    summary["bs_level2"] = bs
    _write_json(summary_path, summary)
    _write_json(metrics / "ir_2_recompute_audit.json", {
        "family": family,
        "recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "scripts.level2.ir.compute_ir2 from prior matched-assets manifest",
        "matched_assets_manifest": str(manifest),
        "num_samples": len(rows),
        "prior_bg_metric": str(bg_path),
        "bg_recomputed": False,
        "ir_2": ir2.value,
        "bs_level2": bs,
    })
    print(json.dumps({"family": family, "num_samples": len(rows), "bg": bg,
                      "ir_2": ir2.value, "bs_level2": bs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
