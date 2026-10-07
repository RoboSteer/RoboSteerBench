#!/usr/bin/env python3
"""Evaluate Level-3 interleaving as duration-weighted Level-2 generation.

Every non-image item in ``input.interleave.sequence`` is a text, audio, or
video generation subtask.  It has no explicit Level-2 modifier, so
``IR_1 = IR_2 = 1`` and its ``BS_level2`` is its modality's BG, or zero for an
explicitly zero-scored missing prediction.  The evaluator
compares the already-segmented Level-3 prediction with the matching temporal
slice of the original full-motion GT.  Image boundary items are recorded but
do not enter the denominator or score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
for import_root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation.model_paths import default_registry


_MODALITIES = ("text", "audio", "video")
_ENCODERS = {"text": "text_motion", "audio": "audio_motion", "video": "video_motion_human"}
_LEVEL_METRIC_ALIASES = {"ir1", "ir2", "ir3", "bs1", "bs2", "bs3", "bslevel1", "bslevel2", "bslevel3"}
_REQUIRED_CSV = ("body_pos.csv", "body_quat.csv", "joint_pos.csv")


@dataclass(frozen=True)
class Segment:
    task_id: str
    source_id: str
    order: int
    modality: str
    start_time: float
    end_time: float
    condition: str
    groundtruth: Path
    task_json: Path

    @property
    def sample_id(self) -> str:
        return f"{self.task_id}__order_{self.order}"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def evaluate_level3(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.evaluation.conventional.runner import read_metric_value, run_conventional
    from scripts.evaluation.shared.task_assets import discover_motion_clips, index_task_records

    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    records = index_task_records(args.task_root, dataset_root)
    segments, ignored_images = _segments_from_records(records, dataset_root)
    if not segments:
        raise RuntimeError("no text/audio/video Level-3 segments found")

    prediction_roots = {
        modality: getattr(args, f"{modality}_prediction")
        for modality in _MODALITIES
        if getattr(args, f"{modality}_prediction") is not None
    }
    if not prediction_roots:
        raise ValueError("provide at least one of --text-prediction, --audio-prediction, --video-prediction")
    zero_modalities = {modality for modality in _MODALITIES if getattr(args, f"missing_{modality}_zero")}
    missing_roots = sorted({segment.modality for segment in segments} - set(prediction_roots) - zero_modalities)
    if missing_roots:
        raise ValueError(f"missing prediction root(s) for Level-3 modalities: {', '.join(missing_roots)}")

    predictions = {
        modality: _match_predictions(discover_motion_clips(path), [segment for segment in segments if segment.modality == modality])
        for modality, path in prediction_roots.items()
    }
    missing_predictions = [segment.sample_id for segment in segments
                           if segment.modality in predictions
                           and segment.sample_id not in predictions[segment.modality]]
    if missing_predictions and not args.missing_prediction_zero:
        preview = ", ".join(missing_predictions[:8])
        raise RuntimeError(f"missing Level-3 prediction for {len(missing_predictions)} task segment(s): {preview}")

    modality_bs: dict[str, float] = {}
    segment_bs: dict[str, float] = {}
    modality_summaries: dict[str, Any] = {}
    all_segment_rows: list[dict[str, Any]] = []
    for modality in _MODALITIES:
        selected = [segment for segment in segments if segment.modality == modality]
        if not selected:
            continue
        if modality not in prediction_roots and modality in zero_modalities:
            modality_bs[modality] = 0.0
            modality_summaries[modality] = {
                "num_segments": len(selected), "bg": None, "bs_level2": 0.0,
                "status": "missing_prediction_zero",
                "policy": "zero_score_keep_duration_weight",
            }
            all_segment_rows.extend(_segment_rows(selected, 0.0))
            continue
        scored = [segment for segment in selected if segment.sample_id in predictions[modality]]
        missing = [segment for segment in selected if segment.sample_id not in predictions[modality]]
        if len(scored) < 2:
            raise RuntimeError(f"Level-3 {modality} has {len(scored)} scored segments; FID requires at least two")
        assets = output / "assets" / modality
        prediction_root = _materialize_predictions(scored, predictions[modality], assets / "prediction")
        groundtruth_root = _materialize_groundtruth_slices(
            scored, assets / "motion_groundtruth", groundtruth_fps=float(config["motion_groundtruth_fps"])
        )
        condition_manifest = _write_condition_manifest(scored, assets / "conditions.json")
        timing_manifest = _write_timing_manifest(scored, assets / "timing.jsonl")
        conventional = run_conventional(
            prediction=prediction_root,
            motion_groundtruth=groundtruth_root,
            instruction_groundtruth=condition_manifest,
            task_metadata=timing_manifest,
            output=output / "modalities" / modality,
            mm_encoder=_ENCODERS[modality],
            metrics=_conventional_metrics(args.metrics),
            models=args.models,
            config_path=args.config,
            device=args.device,
            config_overrides=_runtime_config_overrides(args),
        )
        bg = read_metric_value(output / "modalities" / modality, "BG")
        modality_bs[modality] = bg
        _write_json(output / "modalities" / modality / "metrics" / "bs_level2.json", {
            "name": "BS_level2", "value": bg, "direction": "higher_is_better",
            "formula": "BG × IR_1 × IR_2", "bg": bg, "ir_1": 1.0, "ir_2": 1.0,
            "definition": "Level-3 generation segment has no explicit Level-2 modifier",
        })
        modality_summaries[modality] = {"num_segments": len(selected), "bg": bg, "bs_level2": bg,
                                        "conventional": conventional}
        all_segment_rows.extend(_segment_rows(scored, bg))
        if missing:
            segment_bs.update({segment.sample_id: 0.0 for segment in missing})
            modality_summaries[modality]["num_scored_segments"] = len(scored)
            modality_summaries[modality]["num_missing_predictions"] = len(missing)
            modality_summaries[modality]["missing_prediction_policy"] = "zero_score_keep_duration_weight"
            all_segment_rows.extend(_segment_rows(missing, 0.0))

    samples = _aggregate_samples(segments, modality_bs, segment_bs)
    bs_level3 = sum(float(sample["bs_level3"]) for sample in samples) / len(samples)
    _write_jsonl(output / "level3_segments.jsonl", [*all_segment_rows, *ignored_images])
    _write_jsonl(output / "level3_samples.jsonl", samples)
    metric_payload = {
        "name": "BS_level3", "value": bs_level3, "direction": "higher_is_better",
        "formula": "macro_mean_n(sum_i(duration_n_i / total_duration_n × BS_level2(segment_n_i)))",
        "aggregation": "time_weighted_within_sample_then_macro_mean_over_tasks",
        "ir_1": 1.0, "ir_2": 1.0, "ignored_item_type": "image",
        "num_valid_tasks": len(samples), "num_generation_segments": len(segments),
        "modality_bs_level2": modality_bs,
    }
    zero_scored_modalities = sorted(zero_modalities - set(prediction_roots))
    if zero_scored_modalities:
        metric_payload["zero_scored_modalities"] = zero_scored_modalities
        metric_payload["missing_modality_policy"] = "zero_score_keep_duration_weight"
    if "video" in zero_scored_modalities:
        metric_payload["missing_video_policy"] = "zero_score_keep_duration_weight"
    if missing_predictions:
        metric_payload["missing_prediction_policy"] = "zero_score_keep_duration_weight"
        metric_payload["num_missing_predictions"] = len(missing_predictions)
    _write_json(output / "metrics" / "bs_level3.json", metric_payload)
    result = {"schema": "robosteer.level3.v1", "task_root": str(args.task_root.expanduser().resolve()),
              "prediction_roots": {key: str(value.expanduser().resolve()) for key, value in prediction_roots.items()},
              "modalities": modality_summaries, "num_valid_tasks": len(samples),
              "num_generation_segments": len(segments), "num_ignored_images": len(ignored_images),
              "bs_level3": bs_level3}
    _write_json(output / "level3_summary.json", result)
    return result


def _segments_from_records(records: Iterable[Any], dataset_root: Path) -> tuple[list[Segment], list[dict[str, Any]]]:
    segments: list[Segment] = []
    ignored: list[dict[str, Any]] = []
    for record in records:
        payload = json.loads(record.task_json.read_text(encoding="utf-8"))
        sequence = payload.get("input", {}).get("interleave", {}).get("sequence")
        if not isinstance(sequence, list):
            raise ValueError(f"{record.task_json}: input.interleave.sequence must be a list")
        for item in sequence:
            if not isinstance(item, dict):
                raise ValueError(f"{record.task_json}: interleave item must be an object")
            kind = str(item.get("type", ""))
            order = item.get("order")
            if not isinstance(order, int) or order < 0:
                raise ValueError(f"{record.task_json}: interleave order must be a non-negative integer")
            if kind == "image":
                ignored.append({"task_id": record.task_id, "order": order, "type": "image", "status": "ignored",
                                "reason": "image boundary is excluded from Level-3 duration weighting"})
                continue
            if kind not in _MODALITIES:
                raise ValueError(f"{record.task_json}: unsupported Level-3 item type {kind!r}")
            try:
                start, end = float(item["start_time"]), float(item["end_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{record.task_json}: {kind} item needs numeric start_time/end_time") from exc
            if not 0.0 <= start < end:
                raise ValueError(f"{record.task_json}: invalid interval [{start}, {end}]")
            condition = _condition_path_or_text(item, kind, dataset_root, record.task_json)
            segments.append(Segment(record.task_id, record.sample_id, order, kind, start, end, condition,
                                    record.motion_groundtruth, record.task_json))
    return sorted(segments, key=lambda item: (item.task_id, item.order)), ignored


def _condition_path_or_text(item: dict[str, Any], modality: str, dataset_root: Path, task_json: Path) -> str:
    if modality == "text":
        value = item.get("text")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{task_json}: text segment needs non-empty text")
        return value.strip()
    key = f"{modality}_path"
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{task_json}: {modality} segment needs {key}")
    path = (dataset_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{task_json}: segment condition does not exist: {path}")
    return str(path)


def _match_predictions(candidates: dict[str, Path], segments: Iterable[Segment]) -> dict[str, Path]:
    by_name: dict[str, list[Path]] = {}
    for raw, path in candidates.items():
        normalized = raw.removeprefix("res_").casefold()
        variants = {normalized}
        if raw.startswith("res_") and any(normalized.endswith(f"__{modality}_120") for modality in _MODALITIES):
            variants.add(normalized[:-4])
        for modality in _MODALITIES:
            suffix = f"__{modality}_continuous"
            if normalized.endswith(suffix):
                variants.add(normalized[:-len(suffix)])
        for name in variants:
            by_name.setdefault(name, []).append(path)

    matched: dict[str, Path] = {}
    for segment in segments:
        base = segment.sample_id
        expected = {base.casefold(), f"{base}__{segment.modality}".casefold()}
        hits = {path for name in expected for path in by_name.get(name, ())}
        if len(hits) > 1:
            raise ValueError(f"multiple predictions match {segment.sample_id}: {hits}")
        if hits:
            matched[segment.sample_id] = next(iter(hits))
    return matched


def _materialize_predictions(segments: Iterable[Segment], sources: dict[str, Path], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        target = destination / segment.sample_id
        source = sources[segment.sample_id].resolve()
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                raise FileExistsError(f"refusing to replace existing prediction view {target}")
            continue
        target.symlink_to(source, target_is_directory=True)
    return destination


def _materialize_groundtruth_slices(segments: Iterable[Segment], destination: Path, *, groundtruth_fps: float) -> Path:
    if groundtruth_fps <= 0.0:
        raise ValueError("groundtruth FPS must be positive")
    for segment in segments:
        target = destination / segment.sample_id
        target.mkdir(parents=True, exist_ok=True)
        for filename in _REQUIRED_CSV:
            _slice_csv(segment.groundtruth / filename, target / filename, segment.start_time, segment.end_time, groundtruth_fps)
    return destination


def _slice_csv(source: Path, destination: Path, start: float, end: float, fps: float) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{source}: expected header and at least two frames")
    start_index, end_index = int(round(start * fps)), int(round(end * fps))
    if start_index < 0 or end_index >= len(lines) - 1 or end_index <= start_index:
        raise ValueError(f"{source}: interval [{start}, {end}] is outside the observed GT timeline")
    destination.write_text("\n".join([lines[0], *lines[start_index + 1:end_index + 2]]) + "\n", encoding="utf-8")


def _write_condition_manifest(segments: Iterable[Segment], destination: Path) -> Path:
    selected = list(segments)
    modality = selected[0].modality
    if modality == "text":
        payload: dict[str, Any] = {"schema": "liujiahui.instruction_manifest.v1", "kind": "text",
            "encoder": _ENCODERS[modality], "format": "inline_text_v1",
            "samples": [{"sample_id": segment.sample_id, "text": segment.condition} for segment in selected]}
    else:
        samples = []
        for segment in selected:
            row: dict[str, Any] = {"sample_id": segment.sample_id, "path": segment.condition}
            row["assets"] = {"raw_audio": segment.condition} if modality == "audio" else {"video": [segment.condition]}
            samples.append(row)
        payload = {"schema": "liujiahui.instruction_manifest.v1", "kind": modality,
                   "encoder": _ENCODERS[modality], "format": "resolved_assets_v1", "samples": samples}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _write_timing_manifest(segments: Iterable[Segment], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps({"sample_id": segment.sample_id, "duration": segment.duration})
                                        for segment in segments) + "\n", encoding="utf-8")
    return destination


def _conventional_metrics(groups: list[str]) -> tuple[str, ...]:
    values = [name.strip() for group in groups for name in group.split(",") if name.strip()]
    metrics = tuple(name for name in values if _normal(name) not in _LEVEL_METRIC_ALIASES)
    return tuple(dict.fromkeys((*metrics, "BG")))


def _runtime_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.mm_gpus:
        ids = [item.strip() for item in args.mm_gpus.split(",") if item.strip()]
        if not ids or any(not item.isdigit() for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("--mm-gpus must be unique comma-separated non-negative GPU indices")
        overrides["mm_gpu_devices"] = [f"cuda:{int(item)}" for item in ids]
    if args.video_cache_root is not None:
        overrides["video_local_cache_root"] = str(args.video_cache_root.expanduser().resolve())
    if args.mm_video_decode_workers_per_gpu is not None:
        if args.mm_video_decode_workers_per_gpu <= 0:
            raise ValueError("--mm-video-decode-workers-per-gpu must be positive")
        overrides["mm_video_decode_workers_per_gpu"] = args.mm_video_decode_workers_per_gpu
    return overrides


def _segment_rows(segments: Iterable[Segment], bs_level2: float) -> list[dict[str, Any]]:
    return [{"task_id": segment.task_id, "source_id": segment.source_id, "order": segment.order,
             "sample_id": segment.sample_id, "type": segment.modality, "start_time": segment.start_time,
             "end_time": segment.end_time, "duration": segment.duration, "bs_level2": bs_level2,
             "ir_1": 1.0, "ir_2": 1.0, "groundtruth": str(segment.groundtruth),
             "condition": segment.condition, "task_json": str(segment.task_json)} for segment in segments]


def _aggregate_samples(segments: Iterable[Segment], modality_bs: dict[str, float],
                       segment_bs: dict[str, float] | None = None) -> list[dict[str, Any]]:
    segment_bs = segment_bs or {}
    by_task: dict[str, list[Segment]] = {}
    for segment in segments:
        by_task.setdefault(segment.task_id, []).append(segment)
    rows = []
    for task_id, task_segments in sorted(by_task.items()):
        total = sum(segment.duration for segment in task_segments)
        if total <= 0.0:
            raise ValueError(f"{task_id}: no positive-duration generation segment")
        weighted = sum(segment.duration / total * segment_bs.get(segment.sample_id, modality_bs[segment.modality])
                       for segment in task_segments)
        rows.append({"task_id": task_id, "source_id": task_segments[0].source_id,
                     "num_generation_segments": len(task_segments), "total_generation_duration": total,
                     "bs_level3": weighted,
                     "segments": [{"order": segment.order, "type": segment.modality, "duration": segment.duration,
                                   "time_weight": segment.duration / total,
                                   "bs_level2": segment_bs.get(segment.sample_id, modality_bs[segment.modality])}
                                  for segment in task_segments]})
    return rows


def _normal(name: str) -> str:
    return "".join(character.casefold() for character in name if character.isalnum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-prediction", type=Path)
    parser.add_argument("--audio-prediction", type=Path)
    parser.add_argument("--video-prediction", type=Path)
    parser.add_argument("--missing-text-zero", action="store_true",
                        help="score absent Text predictions as zero while retaining their duration weight")
    parser.add_argument("--missing-audio-zero", action="store_true",
                        help="score absent Audio predictions as zero while retaining their duration weight")
    parser.add_argument("--missing-video-zero", action="store_true",
                        help="score absent Video predictions as zero while retaining their duration weight")
    parser.add_argument("--missing-prediction-zero", action="store_true",
                        help="score missing prediction segments as zero while retaining their duration weight")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--mm-gpus", help="comma-separated GPUs for MM encoding")
    parser.add_argument("--video-cache-root", type=Path)
    parser.add_argument("--mm-video-decode-workers-per-gpu", type=int)
    parser.add_argument("--models", type=Path, default=default_registry(PROJECT_ROOT))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation" / "default.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate_level3(parse_args()), ensure_ascii=False, indent=2))
