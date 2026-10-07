#!/usr/bin/env python3
"""Evaluate Text or Audio Level-2 Order with resumable vLLM replicas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
for import_root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation.conventional.runner import read_metric_value, run_conventional
from scripts.evaluation.model_paths import default_registry
from scripts.evaluation.shared.task_assets import materialize_motion_root
from scripts.level1.evaluate import (
    _conventional_metrics,
    _runtime_config_overrides,
)
from scripts.level2.order_vllm import (
    PROMPT_VERSION,
    PARSER_VERSION,
    OrderTask,
    build_order_tasks,
    load_order_records,
    load_cached_result,
    model_manifest_sha256,
    run_inference,
    start_vllm_servers,
    stop_vllm_servers,
)


_ORDER_LEVEL_METRIC_ALIASES = {"ir2", "bs2", "bslevel2"}


def _order_conventional_metrics(groups: list[str]) -> tuple[str, ...]:
    values = [name.strip() for group in groups for name in group.split(",") if name.strip()]
    conventional = [
        name for name in values
        if "".join(character.casefold() for character in name if character.isalnum())
        not in _ORDER_LEVEL_METRIC_ALIASES
    ]
    return _conventional_metrics(conventional)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _manifest_rows(tasks: list[OrderTask]) -> list[dict[str, Any]]:
    return [{
        "sample_id": task.sample_id,
        "base_sample_id": task.base_sample_id,
        "task_id": task.task_id,
        "task_json": str(task.task_json),
        "prediction_path": None if task.prediction_path is None else str(task.prediction_path),
        "video_path": None if task.video_path is None else str(task.video_path),
        "condition_text": task.condition_text,
        "condition_audio": task.condition_audio,
        "condition_videos": task.condition_videos,
        "motion_groundtruth": task.motion_groundtruth,
        "first_action": task.first_action,
        "second_action": task.second_action,
        "candidate_mapping": {"A": task.action_a, "B": task.action_b},
        "target_first": task.target_first,
    } for task in tasks]


def _ir2_summary(tasks: list[OrderTask], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        result = results[task.sample_id]
        rows.append({
            "level2_task_id": task.task_id,
            "sample_id": task.sample_id,
            "base_sample_id": task.base_sample_id,
            "value": int(result.get("score", 0)),
            "valid": bool(result.get("valid", False)),
            "decoded_video": bool(result.get("decoded_video", False)),
            "a_visible": (result.get("parsed_output") or {}).get("a_visible"),
            "b_visible": (result.get("parsed_output") or {}).get("b_visible"),
            "first": (result.get("parsed_output") or {}).get("first"),
            "target_first": task.target_first,
            "cache_key": result.get("cache_key"),
            "error": result.get("error"),
            "task_family": "Order",
        })
    expected = len(rows)
    successes = sum(row["value"] for row in rows)
    valid = sum(row["valid"] for row in rows)
    decoded = sum(row["decoded_video"] for row in rows)
    both_visible = sum(
        row["valid"] and row["a_visible"] is True and row["b_visible"] is True
        for row in rows
    )
    return {
        "value": successes / expected if expected else 0.0,
        "rows": rows,
        "details": {
            "definition": "binary_closed_set_action_visibility_and_onset_order_macro_mean",
            "task_family": "Order",
            "aggregation": "sum_success_over_complete_expected_task_count",
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "num_expected_tasks": expected,
            "num_decoded_videos": decoded,
            "num_missing_or_corrupt_videos": expected - decoded,
            "num_valid_json": valid,
            "num_invalid_json": decoded - valid,
            "num_both_actions_visible": both_visible,
            "visible_action_pair_rate": both_visible / expected if expected else 0.0,
            "num_successful_constraints": successes,
            "missing_and_invalid_policy": "remain_in_denominator_and_score_zero",
        },
    }


def _materialize_level2_gt_by_task(samples: dict[str, Path], destination: Path) -> Path:
    """Create unique task directories while sharing the converted GT CSV files."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for sample_id, source in sorted(samples.items()):
        source = source.resolve()
        target = destination / sample_id
        if target.is_symlink():
            raise FileExistsError(f"expected a real GT alias directory, found symlink: {target}")
        target.mkdir(exist_ok=True)
        for filename in ("body_pos.csv", "body_quat.csv", "joint_pos.csv"):
            source_file = source / filename
            if not source_file.is_file():
                raise FileNotFoundError(f"converted Level-2 GT lacks {filename}: {source}")
            link = target / filename
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != source_file:
                    raise FileExistsError(f"refusing to replace GT alias {link}")
                continue
            link.symlink_to(source_file)
    return destination


def _resolve_level2_gt(task: OrderTask, dataset_root: Path) -> Path:
    path = Path(task.motion_groundtruth)
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _compute_bg(
    args: argparse.Namespace, tasks: list[OrderTask], output: Path
) -> tuple[dict[str, Any], float, list[dict[str, str]]]:
    """Compute BG from Order's own Level-2 prediction, text, and motion GT."""
    dataset_root = args.dataset_root.expanduser().resolve()
    selected: list[OrderTask] = []
    gt_by_base: dict[str, Path] = {}
    missing: list[dict[str, str]] = []
    for task in tasks:
        if task.prediction_path is None:
            missing.append({
                "level2_task_id": task.task_id, "base_sample_id": task.base_sample_id,
                "missing": "level2_prediction",
            })
            continue
        groundtruth = _resolve_level2_gt(task, dataset_root)
        if not groundtruth.is_file():
            missing.append({
                "level2_task_id": task.task_id, "base_sample_id": task.base_sample_id,
                "missing": "level2_motion_groundtruth",
            })
            continue
        previous = gt_by_base.get(task.base_sample_id)
        if previous is not None and previous != groundtruth:
            raise ValueError(
                f"Order pair {task.base_sample_id} points to conflicting Level-2 GT: "
                f"{previous} and {groundtruth}"
            )
        gt_by_base[task.base_sample_id] = groundtruth
        selected.append(task)
    if not selected:
        raise RuntimeError("no Order task has matching Level-2 prediction and Level-2 GT assets")

    assets = output / "assets"
    gt_manifest = assets / "level2_gt_manifest.jsonl"
    _write_jsonl(gt_manifest, [
        {"sample_id": sample_id, "source": str(path)}
        for sample_id, path in sorted(gt_by_base.items())
    ])
    gt_cache = (
        args.level2_gt_cache.expanduser().resolve()
        if args.level2_gt_cache is not None
        else (output / "cache" / "level2_gt_g1").resolve()
    )
    command = [
        str(args.gmr_python.expanduser().resolve()),
        str((Path(__file__).parent / "retarget_smplx_gt.py").resolve()),
        "--manifest", str(gt_manifest),
        "--output", str(gt_cache),
        "--gmr-root", str(args.gmr_root.expanduser().resolve()),
        "--workers", str(args.gt_retarget_workers),
        "--progress", str(output / "level2_gt_retarget_progress.json"),
    ]
    subprocess.run(command, check=True)

    prediction_root = materialize_motion_root(
        {task.sample_id: task.prediction_path for task in selected if task.prediction_path is not None},
        assets / "level2_prediction",
    )
    gt_root = _materialize_level2_gt_by_task(
        {task.sample_id: gt_cache / task.base_sample_id for task in selected},
        assets / "level2_motion_groundtruth_by_task",
    )
    condition = assets / "level2_conditions.json"
    if args.modality == "audio":
        condition_samples = []
        for task in sorted(selected, key=lambda item: item.sample_id):
            if task.condition_audio is None:
                raise ValueError(f"{task.task_json}: missing Audio Order condition")
            audio = Path(task.condition_audio)
            audio = audio if audio.is_absolute() else dataset_root / audio
            audio = audio.resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"{task.task_json}: audio condition does not exist: {audio}")
            condition_samples.append({
                "sample_id": task.sample_id, "path": str(audio),
                "assets": {"raw_audio": str(audio)},
            })
        condition_payload = {
            "schema": "liujiahui.instruction_manifest.v1",
            "kind": "audio", "encoder": "audio_motion",
            "format": "resolved_assets_v1", "samples": condition_samples,
        }
    elif args.modality == "video":
        condition_samples = []
        for task in sorted(selected, key=lambda item: item.sample_id):
            if len(task.condition_videos) != 2:
                raise ValueError(f"{task.task_json}: Video Order requires two ordered clips")
            videos = [(Path(value) if Path(value).is_absolute() else dataset_root / value).resolve()
                      for value in task.condition_videos]
            for video in videos:
                if not video.is_file():
                    raise FileNotFoundError(f"{task.task_json}: video condition does not exist: {video}")
            condition_samples.append({"sample_id": task.sample_id, "path": str(videos[0]),
                                      "assets": {"video": [str(video) for video in videos]}})
        condition_payload = {
            "schema": "liujiahui.instruction_manifest.v1", "kind": "video",
            "encoder": "video_motion_human", "format": "resolved_assets_v1",
            "condition_aggregation": "ordered_segment_embeddings_l2_normalized_mean_v1",
            "samples": condition_samples,
        }
    else:
        condition_payload = {
            "schema": "liujiahui.instruction_manifest.v1",
            "kind": "text", "encoder": "text_motion",
            "format": "inline_text_v1",
            "samples": [
                {"sample_id": task.sample_id, "text": task.condition_text}
                for task in sorted(selected, key=lambda item: item.sample_id)
            ],
        }
    _write_json(condition, condition_payload)
    timing = assets / "level2_timing.jsonl"
    _write_jsonl(timing, [
        {"sample_id": task.sample_id, "duration": task.duration_seconds}
        for task in sorted(selected, key=lambda item: item.sample_id)
    ])
    summary = run_conventional(
        prediction=prediction_root,
        motion_groundtruth=gt_root,
        instruction_groundtruth=condition,
        task_metadata=timing,
        output=output / "level2_generation",
        mm_encoder=args.mm_encoder,
        metrics=_order_conventional_metrics(args.metrics),
        models=args.models,
        config_path=args.config,
        device=args.device,
        config_overrides=_runtime_config_overrides(args),
    )
    return summary, read_metric_value(output / "level2_generation", "BG"), missing


def evaluate_order(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = load_order_records(args.level2_task_root, args.modality)
    action_records = None
    if args.modality in {"audio", "video"}:
        action_root = args.action_task_root or args.level2_task_root.parent / "text"
        action_records = load_order_records(action_root, "text")
    tasks = build_order_tasks(records, args.rendered_videos, args.prediction, action_records)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if not tasks:
        raise RuntimeError("no Order tasks selected")
    _write_jsonl(output / "assets" / "order_manifest.jsonl", _manifest_rows(tasks))

    model = args.vlm_model.expanduser().resolve()
    model_manifest_hash = model_manifest_sha256(model)
    cache_dir = args.cache_dir.expanduser().resolve()
    servers = []
    endpoints = [item.strip().rstrip("/") for item in args.vlm_endpoints.split(",") if item.strip()] if args.vlm_endpoints else []
    needs_server = any(
        task.video_path is not None and load_cached_result(
            task, cache_dir, model_revision=args.vlm_model_revision,
            model_manifest_sha256_value=model_manifest_hash,
            fps=args.video_fps, max_frames=args.video_max_frames,
            temperature=args.temperature, seed=args.seed,
        ) is None
        for task in tasks
    )
    try:
        if not endpoints and needs_server:
            gpus = [int(item.strip()) for item in args.vlm_gpus.split(",") if item.strip()]
            if not gpus or len(set(gpus)) != len(gpus):
                raise ValueError("--vlm-gpus must contain unique comma-separated GPU indices")
            servers = start_vllm_servers(
                model=model, gpus=gpus, base_port=args.vlm_base_port,
                served_model_name=args.vlm_served_model_name,
                video_root=args.rendered_videos, log_dir=cache_dir / "server_logs",
                max_model_len=args.vlm_max_model_len, max_num_seqs=args.vlm_max_num_seqs,
                gpu_memory_utilization=args.vlm_gpu_memory_utilization,
                startup_timeout=args.vlm_startup_timeout,
            )
            endpoints = [server.endpoint for server in servers]
        results = run_inference(
            tasks, endpoints, cache_dir=cache_dir,
            served_model_name=args.vlm_served_model_name,
            model_revision=args.vlm_model_revision,
            model_manifest_sha256_value=model_manifest_hash,
            fps=args.video_fps, max_frames=args.video_max_frames,
            temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens,
            request_timeout=args.request_timeout, retries=args.request_retries,
            requests_per_server=args.requests_per_server,
            progress_path=output / "order_progress.json",
        )
    finally:
        stop_vllm_servers(servers)

    ir2 = _ir2_summary(tasks, results)
    metrics = output / "metrics"
    _write_json(metrics / "ir_2.json", {
        "name": "IR_2", "value": ir2["value"], "direction": "higher_is_better",
        **ir2["details"],
    })
    _write_jsonl(metrics / "ir_2_samples.jsonl", ir2["rows"])

    conventional_summary = None
    bg = None
    missing_level2_assets: list[dict[str, str]] = []
    bs = None
    if not args.skip_bg:
        conventional_summary, bg, missing_level2_assets = _compute_bg(args, tasks, output)
        bs = bg * ir2["value"]
        _write_json(metrics / "bs_level2.json", {
            "name": "BS_level2", "value": bs, "direction": "higher_is_better",
            "formula": "BG(Level-2 Order prediction vs Level-2 Order GT) × IR_2(Level-2 Order)",
            "bg": bg, "ir_2": ir2["value"], "task_family": "Order",
        })

    summary = {
        "schema": "robosteer.level2.v3",
        "task_family": "Order",
        "modality": args.modality,
        "mm_encoder": args.mm_encoder,
        "prediction": str(args.prediction.expanduser().resolve()),
        "rendered_videos": str(args.rendered_videos.expanduser().resolve()),
        "motion_groundtruth_source": "Level-2 Order task JSON ground_truth.motion_parameters",
        "level2_gt_cache": str(
            args.level2_gt_cache.expanduser().resolve()
            if args.level2_gt_cache is not None else (output / "cache" / "level2_gt_g1").resolve()
        ),
        "vlm_model": str(model),
        "vlm_model_revision": args.vlm_model_revision,
        "vlm_model_manifest_sha256": model_manifest_hash,
        "sampling_config": {"fps": args.video_fps, "max_frames": args.video_max_frames},
        "processor_config": {"cap_pixels_per_frame": True},
        "decoding_config": {"temperature": args.temperature, "do_sample": False, "seed": args.seed},
        "num_level2_predictions": sum(task.prediction_path is not None for task in tasks),
        "num_expected_tasks": len(tasks),
        "num_unique_base_samples": len({task.base_sample_id for task in tasks}),
        "missing_level2_assets": missing_level2_assets,
        "conventional_level2_generation": conventional_summary,
        "bg": bg,
        "ir_2": ir2["value"],
        "ir_2_details": ir2["details"],
        "bs_level2": bs,
    }
    _write_json(output / "level2_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--rendered-videos", type=Path, required=True)
    parser.add_argument("--level2-task-root", type=Path, required=True)
    parser.add_argument("--action-task-root", type=Path,
                        help="Text Order task root supplying action names for Audio tasks")
    parser.add_argument("--modality", choices=("text", "audio", "video"), default="text")
    parser.add_argument("--base-prediction", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--base-motion-groundtruth", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--base-condition-groundtruth", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--level2-gt-cache", type=Path)
    parser.add_argument(
        "--gmr-root", type=Path,
        default=Path(os.environ.get("ROBOOSTEER_GMR_ROOT", PROJECT_ROOT / "third_party" / "GMR")),
    )
    parser.add_argument(
        "--gmr-python", type=Path,
        default=Path(os.environ.get("ROBOOSTEER_GMR_PYTHON", sys.executable)),
    )
    parser.add_argument("--gt-retarget-workers", type=int, default=32)
    parser.add_argument("--vlm-model", type=Path, required=True)
    parser.add_argument("--vlm-model-revision", default="7b87cbf57a58af3c072731d70f7c66e36077e43d")
    parser.add_argument("--vlm-served-model-name", default="qwen3-vl-order-evaluator")
    parser.add_argument("--vlm-gpus", default="0,1,2,3")
    parser.add_argument("--vlm-base-port", type=int, default=18000)
    parser.add_argument("--vlm-endpoints", help="comma-separated pre-launched /v1 endpoints")
    parser.add_argument("--vlm-max-model-len", type=int, default=32768)
    parser.add_argument("--vlm-max-num-seqs", type=int, default=8)
    parser.add_argument("--vlm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vlm-startup-timeout", type=float, default=900)
    parser.add_argument("--requests-per-server", type=int, default=8)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--video-max-frames", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["FID,MM-Distance,BG,IR_2,BS_2"])
    parser.add_argument("--mm-encoder", choices=("text_motion", "audio_motion", "video_motion_human"))
    parser.add_argument("--mm-gpus", default="0,1,2,3")
    parser.add_argument("--video-cache-root", type=Path)
    parser.add_argument("--mm-video-decode-workers-per-gpu", type=int)
    parser.add_argument("--models", type=Path, default=default_registry(PROJECT_ROOT))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation" / "default.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="smoke-test only; evaluate first N tasks")
    parser.add_argument("--skip-bg", action="store_true", help="run only VideoLLM IR_2")
    args = parser.parse_args()
    expected_encoder = {"text": "text_motion", "audio": "audio_motion", "video": "video_motion_human"}[args.modality]
    if args.mm_encoder is None:
        args.mm_encoder = expected_encoder
    elif args.mm_encoder != expected_encoder:
        parser.error(f"--modality {args.modality} requires --mm-encoder {expected_encoder}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.gt_retarget_workers <= 0:
        parser.error("--gt-retarget-workers must be positive")
    if args.requests_per_server <= 0 or args.video_max_frames <= 0 or args.max_tokens <= 0:
        parser.error("concurrency, frame, and token limits must be positive")
    if args.temperature != 0:
        parser.error("the frozen Order protocol requires --temperature 0")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate_order(parse_args()), ensure_ascii=False, indent=2))
