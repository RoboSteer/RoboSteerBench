"""Compute the Level-1 BehaviorSteer score from completed BG results."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math

from pathlib import Path
from typing import Any


IR_1_BY_TASK_TYPE = {
    "motion_generation": 1.0,
}
TEMPORAL_IOU_TASK_TYPES = {"motion_fore", "motion_retro", "motion_inter"}
KEYFRAME_TASK_TYPE = "keyframe_conditioning"
TARGET_REACHING_TASK_TYPE = "target_reaching"
TARGET_REACHING_ANCHOR_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SPATIAL_COMPLETION_JOINT_GROUPS = {
    # Upper-to-Full supplies upper-body motion and requires lower-body completion.
    "upper_full": tuple(range(0, 12)),
    # Lower-to-Full supplies lower-body motion and requires upper-body completion.
    "lower_full": tuple(range(12, 29)),
}
G1_JOINT_ORDER = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
SUPPORTED_TASK_TYPES = set(IR_1_BY_TASK_TYPE) | TEMPORAL_IOU_TASK_TYPES | {KEYFRAME_TASK_TYPE, TARGET_REACHING_TASK_TYPE} | set(SPATIAL_COMPLETION_JOINT_GROUPS)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYFRAME_TASK_ROOT = (
    PROJECT_ROOT.parents[2] / "public" / "Tasks" / "Level1" / "temporal_completion" / "key_frame_conditioning"
)
_KEYFRAME_REFERENCE_CACHE: dict[tuple[str, tuple[float, ...], float, float], Any] = {}
_KEYFRAME_TIME_CACHE: dict[str, tuple[tuple[float, ...], Path]] = {}
_KEYFRAME_SOURCE_TASK_IDS: dict[str, str] | None = None
_KEYFRAME_FK_EVALUATOR: Any | None = None
_TARGET_REACHING_REFERENCE_CACHE: dict[tuple[str, tuple[float, ...], float, float], Any] = {}
_SPATIAL_GT_RANGE_CACHE: dict[str, Any] = {}



def infer_task_type(result_task_dir: Path) -> str | None:
    """Map e.g. ``text_motion_generation`` to ``motion_generation``."""
    name = result_task_dir.name
    for task_type in SUPPORTED_TASK_TYPES:
        if name == task_type or name.endswith(f"_{task_type}"):
            return task_type
    return None


def infer_task_type_from_metrics_dir(metrics_dir: Path) -> str | None:
    """Find the task folder for flat and modality-nested result layouts.

    Most results use ``results/<task>/<method>/metrics``, while video results
    use ``results/<task>/<human|skel>/<method>/metrics``. Searching ancestors
    avoids assuming either fixed depth.
    """
    for candidate in metrics_dir.parents:
        task_type = infer_task_type(candidate)
        if task_type is not None:
            return task_type
    return None


def read_bg(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    if payload.get("name") not in (None, "BG"):
        raise ValueError(f"{path}: expected metric name 'BG', got {payload.get('name')!r}")
    if "value" not in payload:
        raise ValueError(f"{path}: missing numeric 'value' field")
    value = float(payload["value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: BG must be finite and non-negative, got {value!r}")
    return value


def _count_csv_data_rows(path: Path) -> int:
    """Count CSV data rows without parsing numeric values.

    Evaluation exports contain one header and one contiguous data row per
    frame.  Reading once and using bytes.count keeps this post-hoc score from
    spending most of its time in Python line iteration; a trailing NUL pad is
    stripped before the final-row check.
    """
    content = path.read_bytes()
    first_newline = content.find(b"\n")
    if first_newline < 0:
        raise ValueError(f"{path}: missing CSV header newline")
    data = content[first_newline + 1:].rstrip(b"\x00\r\n \t")
    return 0 if not data else data.count(b"\n") + 1


def _prediction_duration_seconds(clip: Path, prediction_fps: float) -> tuple[float, int]:
    if prediction_fps <= 0.0 or not math.isfinite(prediction_fps):
        raise ValueError(f"prediction_fps must be positive and finite, got {prediction_fps!r}")
    frame_counts = [
        _count_csv_data_rows(clip / filename)
        for filename in ("body_pos.csv", "body_quat.csv", "joint_pos.csv")
    ]
    frames = min(frame_counts)
    if frames < 2:
        raise ValueError(f"{clip}: fewer than two common CSV frames: {frame_counts}")
    # This matches the benchmark's observed-CSV timeline: N samples at FPS
    # occupy the interval from t=0 to t=(N-1)/FPS.
    return (frames - 1) / prediction_fps, frames


def temporal_iou_ir_1(run_dir: Path, *, workers: int) -> tuple[float, dict[str, Any]]:
    """Macro-average min/max duration overlap for one completed temporal run."""
    run_config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.jsonl"
    if not run_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{run_dir}: temporal IR_1 requires run_config.json and manifest.jsonl")
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config: dict[str, Any] = json.load(handle)
    try:
        prediction_fps = float(run_config["protocol"]["prediction_fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{run_config_path}: missing protocol.prediction_fps") from exc

    rows: list[tuple[str, float, Path]] = []
    invalid: dict[str, str] = {}
    manifest_samples = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            manifest_samples += 1
            sample_id = f"line_{line_number}"
            try:
                row: dict[str, Any] = json.loads(line)
                sample_id = str(row["sample_id"])
                gt_duration = float(row["task_duration_seconds"])
                if not math.isfinite(gt_duration) or gt_duration <= 0.0:
                    raise ValueError(f"invalid task_duration_seconds {gt_duration!r}")
                prediction_path = row.get("prediction_motion")
                if not prediction_path:
                    raise ValueError("prediction motion is absent from manifest")
                rows.append((sample_id, gt_duration, Path(str(prediction_path))))
            except Exception as exc:
                invalid[sample_id] = str(exc)

    def score_row(row: tuple[str, float, Path]) -> tuple[str, float | None, str | None]:
        sample_id, gt_duration, prediction_path = row
        try:
            prediction_duration, _ = _prediction_duration_seconds(prediction_path, prediction_fps)
            iou = min(prediction_duration, gt_duration) / max(prediction_duration, gt_duration)
            return sample_id, iou, None
        except Exception as exc:
            return sample_id, None, str(exc)

    ious: list[float] = []
    effective_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="temporal-iou") as pool:
        for sample_id, iou, error in pool.map(score_row, rows):
            if error is None:
                assert iou is not None
                ious.append(iou)
            else:
                invalid[sample_id] = error
    if not ious:
        raise RuntimeError(f"{run_dir}: no valid temporal prediction/GT durations")
    details: dict[str, Any] = {
        "definition": "temporal_iou_macro_mean",
        "aggregation": "macro_mean_over_valid_samples",
        "per_sample_formula": "min(prediction_duration_seconds, groundtruth_duration_seconds) / max(prediction_duration_seconds, groundtruth_duration_seconds)",
        "prediction_duration": "(min(common body_pos/body_quat/joint_pos CSV frames) - 1) / prediction_fps",
        "groundtruth_duration": "manifest.task_duration_seconds (metadata.duration)",
        "prediction_fps": prediction_fps,
        "temporal_io_workers": effective_workers,
        "num_manifest_samples": manifest_samples,
        "num_valid_samples": len(ious),
        "num_invalid_samples": len(invalid),
        "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        "run_config": str(run_config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    return float(sum(ious) / len(ious)), details



def _load_joint_positions_29(clip: Path) -> Any:
    """Load only articulated G1 joint angles, which are sufficient for activity IoU."""
    import numpy as np

    path = clip / "joint_pos.csv"
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 29 or len(values) < 2:
        raise ValueError(f"{path}: expected at least two frames of 29 joint angles, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite joint angles")
    return values


def _resample_joint_positions(
    joint_positions: Any, duration_seconds: float, target_fps: float
) -> Any:
    """Place a joint-angle CSV uniformly on its task duration and resample it."""
    import numpy as np

    values = np.asarray(joint_positions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 29 or len(values) < 2:
        raise ValueError(f"joint angles must have shape (T, 29), T >= 2; got {values.shape}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"invalid task duration {duration_seconds!r}")
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"invalid activity FPS {target_fps!r}")
    source_times = np.linspace(0.0, duration_seconds, len(values), dtype=np.float64)
    target_times = np.arange(
        int(math.floor(duration_seconds * target_fps + 1e-6)) + 1, dtype=np.float64
    ) / target_fps
    output = np.empty((len(target_times), 29), dtype=np.float32)
    for index in range(29):
        output[:, index] = np.interp(target_times, source_times, values[:, index])
    return output


def _joint_activity_mask(joint_positions: Any, *, fps: float, threshold_rad_per_sec: float) -> Any:
    """Boolean active-joint mask for each frame interval and G1 DOF."""
    import numpy as np

    if not math.isfinite(threshold_rad_per_sec) or threshold_rad_per_sec < 0.0:
        raise ValueError(f"activity threshold must be finite and non-negative, got {threshold_rad_per_sec!r}")
    values = np.asarray(joint_positions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 29 or len(values) < 2:
        raise ValueError(f"joint positions must have shape (T, 29), T >= 2; got {values.shape}")
    velocity = np.abs(np.diff(values, axis=0)) * float(fps)
    return velocity >= float(threshold_rad_per_sec)


def _groundtruth_joint_activity(
    groundtruth_path: Path,
    *,
    duration_seconds: float,
    activity_fps: float,
    threshold_rad_per_sec: float,
) -> Any:
    """Load and cache GT activity masks shared by all spatial-completion runs."""
    key = (
        str(groundtruth_path.resolve()),
        float(duration_seconds),
        float(activity_fps),
        float(threshold_rad_per_sec),
        29.0,
    )
    cached = _SPATIAL_GT_ACTIVITY_CACHE.get(key)
    if cached is None:
        q = _load_joint_positions_29(groundtruth_path)
        cached = _joint_activity_mask(
            _resample_joint_positions(q, duration_seconds, activity_fps),
            fps=activity_fps,
            threshold_rad_per_sec=threshold_rad_per_sec,
        )
        _SPATIAL_GT_ACTIVITY_CACHE[key] = cached
    return cached


def spatial_joint_range_overlap_ir_1(
    run_dir: Path,
    *,
    task_type: str,
    workers: int,
    static_range_epsilon_rad: float,
    static_position_tolerance_rad: float,
) -> tuple[float, dict[str, Any]]:
    """Macro-average completed-body joint-angle-range overlap for spatial completion."""
    import numpy as np

    try:
        joint_indices = SPATIAL_COMPLETION_JOINT_GROUPS[task_type]
    except KeyError as exc:
        raise ValueError(f"unsupported spatial task type {task_type!r}") from exc
    if not math.isfinite(static_range_epsilon_rad) or static_range_epsilon_rad < 0.0:
        raise ValueError(f"static range epsilon must be finite and non-negative, got {static_range_epsilon_rad!r}")
    if not math.isfinite(static_position_tolerance_rad) or static_position_tolerance_rad < 0.0:
        raise ValueError(f"static position tolerance must be finite and non-negative, got {static_position_tolerance_rad!r}")

    run_config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.jsonl"
    if not run_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{run_dir}: spatial joint range overlap requires run_config.json and manifest.jsonl")
    default_gt_root = PROJECT_ROOT / "groundtruth" / "motion_generation"

    rows: list[tuple[str, Path, Path]] = []
    excluded: dict[str, str] = {}
    manifest_samples = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        manifest_samples += 1
        sample_id = f"line_{line_number}"
        try:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            prediction_value = row.get("prediction_motion")
            if not prediction_value:
                raise ValueError("prediction motion is absent from manifest")
            gt_value = row.get("motion_groundtruth")
            gt_path = Path(str(gt_value)) if gt_value else default_gt_root / sample_id
            rows.append((sample_id, Path(str(prediction_value)), gt_path))
        except Exception as exc:
            excluded[sample_id] = str(exc)

    selected = np.asarray(joint_indices, dtype=np.int64)

    def groundtruth_ranges(path: Path) -> Any:
        key = str(path.resolve())
        cached = _SPATIAL_GT_RANGE_CACHE.get(key)
        if cached is None:
            q = _load_joint_positions_29(path)
            cached = np.stack((q.min(axis=0), q.max(axis=0)), axis=1)
            _SPATIAL_GT_RANGE_CACHE[key] = cached
        return cached

    def score_row(row: tuple[str, Path, Path]) -> tuple[str, float | None, int, str | None]:
        sample_id, prediction_path, groundtruth_path = row
        try:
            gt_range = groundtruth_ranges(groundtruth_path)[selected]
            pred_q = _load_joint_positions_29(prediction_path)
            pred_range = np.stack((pred_q.min(axis=0), pred_q.max(axis=0)), axis=1)[selected]
            gt_width = gt_range[:, 1] - gt_range[:, 0]
            pred_width = pred_range[:, 1] - pred_range[:, 0]
            overlap = np.maximum(0.0, np.minimum(gt_range[:, 1], pred_range[:, 1]) - np.maximum(gt_range[:, 0], pred_range[:, 0]))
            # GT is the required range. When prediction entirely contains it,
            # its own width becomes the denominator and excess amplitude is penalized.
            prediction_contains_gt = (pred_range[:, 0] <= gt_range[:, 0]) & (pred_range[:, 1] >= gt_range[:, 1])
            denominator = np.where(prediction_contains_gt, pred_width, gt_width)
            scores = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0.0)
            both_static = (gt_width <= static_range_epsilon_rad) & (pred_width <= static_range_epsilon_rad)
            same_static_pose = np.abs(gt_range.mean(axis=1) - pred_range.mean(axis=1)) <= static_position_tolerance_rad
            scores = np.where(both_static, same_static_pose.astype(np.float32), scores)
            if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
                raise ValueError("non-finite or out-of-range joint range-overlap score")
            return sample_id, float(scores.mean()), int(len(scores)), None
        except Exception as exc:
            return sample_id, None, 0, str(exc)

    ious: list[float] = []
    total_scored_joints = 0
    effective_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="spatial-joint-range-overlap") as pool:
        for sample_id, iou, scored_joints, error in pool.map(score_row, rows):
            if error is None:
                assert iou is not None
                ious.append(iou)
                total_scored_joints += scored_joints
            else:
                excluded[sample_id] = error
    if not ious:
        raise RuntimeError(f"{run_dir}: no valid spatial-completion prediction/GT pairs")

    completion_part = "lower" if task_type == "upper_full" else "upper"
    details: dict[str, Any] = {
        "definition": "completed_body_joint_angle_range_overlap",
        "aggregation": "macro_mean_over_valid_samples",
        "completion_part": completion_part,
        "evaluated_joint_indices": list(joint_indices),
        "evaluated_joint_names": [G1_JOINT_ORDER[index] for index in joint_indices],
        "per_joint_range": "[min_t joint_pos[t,j], max_t joint_pos[t,j]] in radians",
        "per_joint_overlap": "max(0, min(gt_max, pred_max) - max(gt_min, pred_min))",
        "per_joint_denominator": "prediction range width when prediction range contains GT range; otherwise GT range width",
        "per_joint_score": "overlap / denominator",
        "per_sample_ir_1": "mean per-joint range-overlap score over the completed-body joints",
        "both_static_joint_score": "1 if both range widths <= static_range_epsilon_rad and mean angles differ by <= static_position_tolerance_rad; otherwise 0",
        "static_range_epsilon_rad": float(static_range_epsilon_rad),
        "static_position_tolerance_rad": float(static_position_tolerance_rad),
        "joint_angle_time_axis": "all joint_pos.csv frames in each clip",
        "num_manifest_samples": manifest_samples,
        "num_valid_samples": len(ious),
        "num_excluded_samples": len(excluded),
        "excluded_samples": dict(list(sorted(excluded.items()))[:32]),
        "total_scored_joints": total_scored_joints,
        "spatial_iou_workers": effective_workers,
        "run_config": str(run_config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    return float(sum(ious) / len(ious)), details


def _keyframe_source_task_id(sample_id: str) -> str:
    """Resolve a benchmark ID once, rather than reparsing the large manifest per clip."""
    global _KEYFRAME_SOURCE_TASK_IDS
    if _KEYFRAME_SOURCE_TASK_IDS is None:
        text_manifest = PROJECT_ROOT / "data" / "text" / "kc" / "manifest.json"
        payload = json.loads(text_manifest.read_text(encoding="utf-8"))
        _KEYFRAME_SOURCE_TASK_IDS = {
            str(row["sample_id"]): str(row["source_task_id"])
            for row in payload.get("samples", [])
            if row.get("sample_id") is not None and row.get("source_task_id") is not None
        }
    try:
        return _KEYFRAME_SOURCE_TASK_IDS[sample_id]
    except KeyError as exc:
        raise KeyError(f"data/text/kc/manifest.json: no key-frame task entry for {sample_id!r}") from exc


def _keyframe_times(sample_id: str, task_root: Path) -> tuple[tuple[float, ...], Path]:
    """Load sparse anchor times from the original Key-Frame task JSON."""
    cached = _KEYFRAME_TIME_CACHE.get(sample_id)
    if cached is not None:
        return cached
    source_task_id = _keyframe_source_task_id(sample_id)
    for task_type in ("IMG_TXT_SKEL", "IMG_TXT_HUMAN"):
        candidate = task_root / task_type / f"{source_task_id}.json"
        if candidate.is_file():
            task_path = candidate
            break
    else:
        raise FileNotFoundError(f"key-frame task JSON is missing for {source_task_id!r} under {task_root}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    images = task.get("input", {}).get("modalities", {}).get("input_images", [])
    times = tuple(float(item["time"]) for item in images)
    duration = float(task.get("metadata", {}).get("duration"))
    if not times or any(not math.isfinite(value) or value < 0.0 or value > duration for value in times):
        raise ValueError(f"{task_path}: invalid input_images key-frame times")
    result = (times, task_path)
    _KEYFRAME_TIME_CACHE[sample_id] = result
    return result


def _qpos_at_times(qpos: Any, duration_seconds: float, times: tuple[float, ...]) -> Any:
    """Interpolate a qpos trajectory on the task-defined [0, duration] timeline."""
    import numpy as np
    from scripts.evaluation.preprocessing.temporal import _slerp_times

    values = np.asarray(qpos, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2:
        raise ValueError(f"qpos must have shape (T, 36), T >= 2; got {values.shape}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"invalid task duration {duration_seconds!r}")
    target = np.asarray(times, dtype=np.float64)
    source = np.linspace(0.0, duration_seconds, len(values), dtype=np.float64)
    output = np.empty((len(target), 36), dtype=np.float32)
    columns = list(range(3)) + list(range(7, 36))
    for column in columns:
        output[:, column] = np.interp(target, source, values[:, column])
    output[:, 3:7] = _slerp_times(source, values[:, 3:7], target)
    return output


def _keyframe_reference(
    sample_id: str,
    groundtruth_path: Path,
    duration_seconds: float,
    groundtruth_fps: float,
    task_root: Path,
) -> tuple[Any, tuple[float, ...], Path]:
    """Return cached GT G1 qpos at the sparse condition timestamps."""
    from scripts.evaluation.data.motion import load_qpos_36

    times, task_path = _keyframe_times(sample_id, task_root)
    key = (str(groundtruth_path.resolve()), times, float(duration_seconds), float(groundtruth_fps))
    cached = _KEYFRAME_REFERENCE_CACHE.get(key)
    if cached is None:
        cached = _qpos_at_times(load_qpos_36(groundtruth_path), duration_seconds, times)
        _KEYFRAME_REFERENCE_CACHE[key] = cached
    return cached, times, task_path


def _root_aligned_keyframe_errors(prediction: Any, groundtruth: Any, *, batch_size: int) -> Any:
    """Return one root-aligned 14-link FK MPJPE value in metres per key frame."""
    import numpy as np
    from scripts.evaluation.core_metric.MPJPE import RootAlignedMPJPEEvaluator

    global _KEYFRAME_FK_EVALUATOR
    if _KEYFRAME_FK_EVALUATOR is None:
        _KEYFRAME_FK_EVALUATOR = RootAlignedMPJPEEvaluator(device="auto")
    evaluator = _KEYFRAME_FK_EVALUATOR
    torch = evaluator.torch
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(groundtruth, dtype=np.float32)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 36:
        raise ValueError(f"key-frame qpos shape mismatch: prediction={pred.shape}, groundtruth={gt.shape}")
    errors: list[Any] = []
    body_indices = torch.as_tensor(evaluator.body_indices, device=evaluator.device)
    for start in range(0, len(pred), max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), len(pred))
        with torch.inference_mode():
            qpos = torch.as_tensor(np.stack((pred[start:end], gt[start:end])), dtype=torch.float32, device=evaluator.device)
            body_pos_w = evaluator.kinematics.forward_kinematics(qpos)["body_pos_w"]
            tracked = body_pos_w.index_select(dim=2, index=body_indices)
            root = tracked[:, :, evaluator.root_link_index : evaluator.root_link_index + 1]
            aligned = tracked - root
            errors.append(torch.linalg.vector_norm(aligned[0] - aligned[1], dim=-1).mean(dim=1).cpu())
    return torch.cat(errors).numpy().astype(np.float32)


def keyframe_ir_1(
    run_dir: Path,
    *,
    workers: int,
    fk_batch_size: int,
    tau_meters: float,
    task_root: Path,
    prediction_window_frames: int,
) -> tuple[float, dict[str, Any]]:
    """Score sparse key-frame pose satisfaction with an exponential FK kernel."""
    import numpy as np
    from scripts.evaluation.data.motion import load_qpos_36

    if tau_meters <= 0.0 or not math.isfinite(tau_meters):
        raise ValueError(f"key-frame tau must be positive and finite, got {tau_meters!r}")
    window_frames = int(prediction_window_frames)
    if window_frames < 1 or window_frames % 2 == 0:
        raise ValueError(f"key-frame prediction window must be a positive odd number, got {prediction_window_frames!r}")
    run_config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.jsonl"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    protocol = run_config.get("protocol", {})
    prediction_fps = float(protocol["prediction_fps"])
    if not math.isfinite(prediction_fps) or prediction_fps <= 0.0:
        raise ValueError(f"{run_config_path}: invalid protocol.prediction_fps {prediction_fps!r}")
    groundtruth_fps = float(protocol.get("motion_groundtruth_fps", prediction_fps))
    default_gt_root = PROJECT_ROOT / "groundtruth" / "motion_generation"
    half_window = window_frames // 2
    window_offsets_seconds = tuple(offset / prediction_fps for offset in range(-half_window, half_window + 1))

    rows: list[tuple[str, Path, Path, float]] = []
    invalid: dict[str, str] = {}
    manifest_samples = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        manifest_samples += 1
        sample_id = f"line_{line_number}"
        try:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            prediction_value = row.get("prediction_motion")
            if not prediction_value:
                raise ValueError("prediction motion is absent from manifest")
            duration = float(row["task_duration_seconds"])
            gt_value = row.get("motion_groundtruth")
            groundtruth_path = Path(str(gt_value)) if gt_value else default_gt_root / sample_id
            rows.append((sample_id, Path(str(prediction_value)), groundtruth_path, duration))
        except Exception as exc:
            invalid[sample_id] = str(exc)

    # Build the sample-to-source-task mapping once before launching concurrent
    # loaders; otherwise every worker races to parse the same large manifest.
    if rows:
        _keyframe_source_task_id(rows[0][0])

    def load_row(row: tuple[str, Path, Path, float]) -> tuple[str, Any | None, Any | None, str | None]:
        sample_id, prediction_path, groundtruth_path, duration = row
        try:
            gt_keyframes, times, _ = _keyframe_reference(
                sample_id, groundtruth_path, duration, groundtruth_fps, task_root
            )
            # A five-frame window is centred on the GT key-frame time. At a
            # clip boundary, clamping keeps the centre condition valid while
            # allowing repeated boundary samples rather than dropping a key frame.
            window_times = tuple(
                min(duration, max(0.0, time + offset))
                for time in times
                for offset in window_offsets_seconds
            )
            prediction_window = _qpos_at_times(
                load_qpos_36(prediction_path), duration, window_times
            )
            groundtruth_repeated = np.repeat(gt_keyframes, window_frames, axis=0)
            return sample_id, prediction_window, groundtruth_repeated, None
        except Exception as exc:
            return sample_id, None, None, str(exc)

    loaded: list[tuple[str, Any, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="keyframe-load") as pool:
        for sample_id, prediction, groundtruth, error in pool.map(load_row, rows):
            if error is None:
                loaded.append((sample_id, prediction, groundtruth))
            else:
                invalid[sample_id] = error
    if not loaded:
        raise RuntimeError(f"{run_dir}: no valid key-frame prediction/GT pairs")

    prediction_all = np.concatenate([prediction for _, prediction, _ in loaded], axis=0)
    groundtruth_all = np.concatenate([groundtruth for _, _, groundtruth in loaded], axis=0)
    window_errors_m = _root_aligned_keyframe_errors(
        prediction_all, groundtruth_all, batch_size=fk_batch_size
    )
    if len(window_errors_m) % window_frames != 0:
        raise RuntimeError(f"{run_dir}: malformed key-frame prediction-window error count")
    # Each consecutive window belongs to one GT key-frame; the best temporal
    # alignment inside the centred prediction window realizes that key frame.
    errors_m = window_errors_m.reshape(-1, window_frames).min(axis=1)
    scores = np.exp(-errors_m / float(tau_meters))
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{run_dir}: non-finite key-frame scores")
    details: dict[str, Any] = {
        "definition": "root_aligned_g1_fk_keyframe_window_match",
        "aggregation": "macro_mean_over_valid_keyframes",
        "per_keyframe_error": "minimum root-aligned mean 14-link G1 FK position error in metres over the centred prediction window",
        "per_keyframe_score": "exp(-minimum_window_keyframe_error_meters / keyframe_tau_meters)",
        "keyframe_tau_meters": float(tau_meters),
        "prediction_window_frames": window_frames,
        "prediction_window_offsets_seconds": list(window_offsets_seconds),
        "prediction_window_boundary_policy": "clamp_to_[0, task_duration_seconds]",
        "prediction_time_axis": "CSV frames distributed uniformly over manifest.task_duration_seconds",
        "groundtruth_time_axis": "CSV frames distributed uniformly over manifest.task_duration_seconds",
        "keyframe_time_source": "original Key-Frame Conditioning Task JSON input.modalities.input_images[].time",
        "num_manifest_samples": manifest_samples,
        "num_valid_samples": len(loaded),
        "num_valid_keyframes": int(len(scores)),
        "num_invalid_samples": len(invalid),
        "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        "keyframe_load_workers": max(1, int(workers)),
        "fk_batch_size": max(1, int(fk_batch_size)),
        "run_config": str(run_config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "keyframe_task_root": str(task_root.resolve()),
    }
    return float(scores.mean()), details



def target_reaching_ir_1(
    run_dir: Path,
    *,
    workers: int,
    fk_batch_size: int,
    tau_meters: float,
    prediction_window_frames: int,
) -> tuple[float, dict[str, Any]]:
    """Uniform four-phase full-body window matching for Target Reaching."""
    import numpy as np
    from scripts.evaluation.data.motion import load_qpos_36

    if tau_meters <= 0.0 or not math.isfinite(tau_meters):
        raise ValueError(f"target-reaching tau must be positive and finite, got {tau_meters!r}")
    window_frames = int(prediction_window_frames)
    if window_frames < 1 or window_frames % 2 == 0:
        raise ValueError(
            f"target-reaching prediction window must be a positive odd number, got {prediction_window_frames!r}"
        )
    run_config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.jsonl"
    if not run_config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{run_dir}: target-reaching IR_1 requires run_config.json and manifest.jsonl")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    protocol = run_config.get("protocol", {})
    prediction_fps = float(protocol["prediction_fps"])
    if not math.isfinite(prediction_fps) or prediction_fps <= 0.0:
        raise ValueError(f"{run_config_path}: invalid protocol.prediction_fps {prediction_fps!r}")
    groundtruth_fps = float(protocol.get("motion_groundtruth_fps", prediction_fps))
    default_gt_root = PROJECT_ROOT / "groundtruth" / "motion_generation"
    half_window = window_frames // 2
    window_offsets_seconds = tuple(offset / prediction_fps for offset in range(-half_window, half_window + 1))

    rows: list[tuple[str, Path, Path, float]] = []
    excluded: dict[str, str] = {}
    manifest_samples = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        manifest_samples += 1
        sample_id = f"line_{line_number}"
        try:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            prediction_value = row.get("prediction_motion")
            if not prediction_value:
                raise ValueError("prediction motion is absent from manifest")
            duration = float(row["task_duration_seconds"])
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"invalid task_duration_seconds {duration!r}")
            gt_value = row.get("motion_groundtruth")
            gt_path = Path(str(gt_value)) if gt_value else default_gt_root / sample_id
            rows.append((sample_id, Path(str(prediction_value)), gt_path, duration))
        except Exception as exc:
            excluded[sample_id] = str(exc)

    def gt_anchors(groundtruth_path: Path, duration: float) -> tuple[Any, tuple[float, ...]]:
        anchor_times = tuple(duration * fraction for fraction in TARGET_REACHING_ANCHOR_FRACTIONS)
        key = (str(groundtruth_path.resolve()), anchor_times, float(duration), float(groundtruth_fps))
        cached = _TARGET_REACHING_REFERENCE_CACHE.get(key)
        if cached is None:
            cached = _qpos_at_times(load_qpos_36(groundtruth_path), duration, anchor_times)
            _TARGET_REACHING_REFERENCE_CACHE[key] = cached
        return cached, anchor_times

    def load_row(row: tuple[str, Path, Path, float]) -> tuple[str, Any | None, Any | None, str | None]:
        sample_id, prediction_path, groundtruth_path, duration = row
        try:
            gt, anchor_times = gt_anchors(groundtruth_path, duration)
            prediction_times = tuple(
                min(duration, max(0.0, time + offset))
                for time in anchor_times
                for offset in window_offsets_seconds
            )
            prediction_window = _qpos_at_times(
                load_qpos_36(prediction_path), duration, prediction_times
            )
            return sample_id, prediction_window, np.repeat(gt, window_frames, axis=0), None
        except Exception as exc:
            return sample_id, None, None, str(exc)

    loaded: list[tuple[str, Any, Any]] = []
    effective_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="target-reaching-load") as pool:
        for sample_id, prediction, groundtruth, error in pool.map(load_row, rows):
            if error is None:
                loaded.append((sample_id, prediction, groundtruth))
            else:
                excluded[sample_id] = error
    if not loaded:
        raise RuntimeError(f"{run_dir}: no valid Target Reaching prediction/GT pairs")

    prediction_all = np.concatenate([prediction for _, prediction, _ in loaded], axis=0)
    groundtruth_all = np.concatenate([groundtruth for _, _, groundtruth in loaded], axis=0)
    window_errors_m = _root_aligned_keyframe_errors(
        prediction_all, groundtruth_all, batch_size=fk_batch_size
    )
    expected_anchors = len(TARGET_REACHING_ANCHOR_FRACTIONS)
    group_size = expected_anchors * window_frames
    if len(window_errors_m) != len(loaded) * group_size:
        raise RuntimeError(f"{run_dir}: malformed target-reaching window error count")
    anchor_errors_m = window_errors_m.reshape(-1, expected_anchors, window_frames).min(axis=2)
    sample_scores = np.exp(-anchor_errors_m / float(tau_meters)).mean(axis=1)
    if not np.isfinite(sample_scores).all():
        raise RuntimeError(f"{run_dir}: non-finite Target Reaching scores")
    details: dict[str, Any] = {
        "definition": "uniform_four_phase_root_aligned_g1_fk_window_match",
        "aggregation": "macro_mean_over_valid_samples",
        "anchor_fractions_of_task_duration": list(TARGET_REACHING_ANCHOR_FRACTIONS),
        "per_anchor_error": "minimum root-aligned mean 14-link G1 FK position error in metres over the centred prediction window",
        "per_anchor_score": "exp(-minimum_window_anchor_error_meters / target_reaching_tau_meters)",
        "per_sample_ir_1": "mean of the four anchor scores",
        "target_reaching_tau_meters": float(tau_meters),
        "prediction_window_frames": window_frames,
        "prediction_window_offsets_seconds": list(window_offsets_seconds),
        "prediction_window_boundary_policy": "clamp_to_[0, task_duration_seconds]",
        "prediction_and_groundtruth_time_axis": "CSV frames distributed uniformly over manifest.task_duration_seconds",
        "num_manifest_samples": manifest_samples,
        "num_valid_samples": len(loaded),
        "num_excluded_samples": len(excluded),
        "excluded_samples": dict(list(sorted(excluded.items()))[:32]),
        "target_reaching_load_workers": effective_workers,
        "fk_batch_size": max(1, int(fk_batch_size)),
        "run_config": str(run_config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    return float(sample_scores.mean()), details


def score_one(
    bg_path: Path, *, overwrite: bool, temporal_workers: int, keyframe_workers: int,
    keyframe_fk_batch_size: int, keyframe_tau_meters: float, keyframe_task_root: Path,
    keyframe_window_frames: int,
    target_reaching_workers: int, target_reaching_fk_batch_size: int,
    target_reaching_tau_meters: float, target_reaching_window_frames: int,
    spatial_iou_workers: int, spatial_static_range_epsilon_rad: float,
    spatial_static_position_tolerance_rad: float,
) -> dict[str, Any] | None:
    metrics_dir = bg_path.parent
    output_path = metrics_dir / "bs_level1.json"
    task_type = infer_task_type_from_metrics_dir(metrics_dir)
    if task_type is None:
        return None
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "task_type": task_type,
            "path": str(output_path),
        }

    bg = read_bg(bg_path)
    if task_type in IR_1_BY_TASK_TYPE:
        ir_1 = IR_1_BY_TASK_TYPE[task_type]
        ir_1_details: dict[str, Any] = {"definition": "constant", "value": ir_1}
    elif task_type == KEYFRAME_TASK_TYPE:
        ir_1, ir_1_details = keyframe_ir_1(
            metrics_dir.parent,
            workers=keyframe_workers,
            fk_batch_size=keyframe_fk_batch_size,
            tau_meters=keyframe_tau_meters,
            task_root=keyframe_task_root,
            prediction_window_frames=keyframe_window_frames,
        )
    elif task_type == TARGET_REACHING_TASK_TYPE:
        ir_1, ir_1_details = target_reaching_ir_1(
            metrics_dir.parent,
            workers=target_reaching_workers,
            fk_batch_size=target_reaching_fk_batch_size,
            tau_meters=target_reaching_tau_meters,
            prediction_window_frames=target_reaching_window_frames,
        )
    elif task_type in SPATIAL_COMPLETION_JOINT_GROUPS:
        ir_1, ir_1_details = spatial_joint_range_overlap_ir_1(
            metrics_dir.parent,
            task_type=task_type,
            workers=spatial_iou_workers,
            static_range_epsilon_rad=spatial_static_range_epsilon_rad,
            static_position_tolerance_rad=spatial_static_position_tolerance_rad,
        )
    else:
        ir_1, ir_1_details = temporal_iou_ir_1(metrics_dir.parent, workers=temporal_workers)
    value = bg * ir_1
    result = {
        "name": "BS_level1",
        "value": value,
        "direction": "higher_is_better",
        "formula": "BG * IR_1",
        "task_type": task_type,
        "bg": bg,
        "ir_1": ir_1,
        "ir_1_details": ir_1_details,
        "source_metrics": {"bg": str(bg_path.resolve())},
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {"status": "written", "task_type": task_type, "path": str(output_path), "value": value, "ir_1": ir_1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BS_level1 = BG * IR_1 for supported Level-1 task types."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing task result folders (default: results).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing metrics/bs_level1.json files.",
    )
    parser.add_argument(
        "--temporal-workers",
        type=int,
        default=32,
        help="Concurrent CSV readers for temporal IoU (default: 32).",
    )
    parser.add_argument("--keyframe-workers", type=int, default=32, help="Concurrent key-frame CSV loaders (default: 32).")
    parser.add_argument("--keyframe-fk-batch-size", type=int, default=4096, help="G1 FK batch size for key-frame IR_1 (default: 4096).")
    parser.add_argument("--keyframe-tau-meters", type=float, default=0.20, help="Exponential key-frame score scale in metres (default: 0.20).")
    parser.add_argument("--keyframe-window-frames", type=int, default=5, help="Odd prediction-frame window centred on each GT key frame (default: 5).")
    parser.add_argument("--keyframe-task-root", type=Path, default=DEFAULT_KEYFRAME_TASK_ROOT, help="Original IMG_TXT_SKEL/HUMAN task JSON root.")
    parser.add_argument("--target-reaching-workers", type=int, default=32, help="Concurrent CSV readers for Target Reaching IR_1 (default: 32).")
    parser.add_argument("--target-reaching-fk-batch-size", type=int, default=4096, help="G1 FK batch size for Target Reaching IR_1 (default: 4096).")
    parser.add_argument("--target-reaching-tau-meters", type=float, default=0.20, help="Exponential Target Reaching score scale in metres (default: 0.20).")
    parser.add_argument("--target-reaching-window-frames", type=int, default=5, help="Odd prediction-frame window centred on each uniform Target Reaching anchor (default: 5).")
    parser.add_argument("--spatial-iou-workers", type=int, default=32, help="Concurrent joint CSV readers for spatial-completion IoU (default: 32).")
    parser.add_argument("--spatial-static-range-epsilon-rad", type=float, default=1e-4, help="Maximum range width considered static for spatial range overlap (default: 1e-4 rad).")
    parser.add_argument("--spatial-static-position-tolerance-rad", type=float, default=0.05, help="Maximum mean-angle difference for matching static joints (default: 0.05 rad).")
    parser.add_argument("--task-types", nargs="+", choices=sorted(SUPPORTED_TASK_TYPES), help="Only score these task types.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise NotADirectoryError(f"results root does not exist: {results_root}")

    selected_task_types = None if args.task_types is None else set(args.task_types)
    summaries = []
    for bg_path in sorted(results_root.glob("*/**/metrics/bg.json")):
        task_type = infer_task_type_from_metrics_dir(bg_path.parent)
        if selected_task_types is not None and task_type not in selected_task_types:
            continue
        summary = score_one(
            bg_path,
            overwrite=args.overwrite,
            temporal_workers=args.temporal_workers,
            keyframe_workers=args.keyframe_workers,
            keyframe_fk_batch_size=args.keyframe_fk_batch_size,
            keyframe_tau_meters=args.keyframe_tau_meters,
            keyframe_task_root=args.keyframe_task_root.expanduser().resolve(),
            keyframe_window_frames=args.keyframe_window_frames,
            target_reaching_workers=args.target_reaching_workers,
            target_reaching_fk_batch_size=args.target_reaching_fk_batch_size,
            target_reaching_tau_meters=args.target_reaching_tau_meters,
            target_reaching_window_frames=args.target_reaching_window_frames,
            spatial_iou_workers=args.spatial_iou_workers,
            spatial_static_range_epsilon_rad=args.spatial_static_range_epsilon_rad,
            spatial_static_position_tolerance_rad=args.spatial_static_position_tolerance_rad,
        )
        if summary is not None:
            summaries.append(summary)

    print(json.dumps({
        "formula": "BS_level1 = BG * IR_1",
        "supported_task_types": {
            **IR_1_BY_TASK_TYPE,
            **{task_type: "temporal_iou_macro_mean" for task_type in sorted(TEMPORAL_IOU_TASK_TYPES)},
            KEYFRAME_TASK_TYPE: "root_aligned_g1_fk_keyframe_window_match_5frames_exp_tau_0.20m",
            TARGET_REACHING_TASK_TYPE: "uniform_four_phase_root_aligned_g1_fk_window_match_5frames_exp_tau_0.20m",
            **{task_type: "completed_body_joint_angle_range_overlap" for task_type in sorted(SPATIAL_COMPLETION_JOINT_GROUPS)},
        },
        "results": summaries,
        "written": sum(item["status"] == "written" for item in summaries),
        "skipped": sum(item["status"] == "skipped" for item in summaries),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
