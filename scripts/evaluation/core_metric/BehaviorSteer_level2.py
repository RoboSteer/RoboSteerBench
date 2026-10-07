"""Level-2 BehaviorSteer instruction-response scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.core_metric.g_MPJPE import GlobalMPJPEEvaluator
from scripts.evaluation.data.motion import MotionIndex, load_qpos_36


@dataclass(frozen=True)
class IR2Score:
    value: float
    sample_ids: list[str]
    sample_values: np.ndarray
    details: dict[str, Any]


def speed_ir_2(
    prediction: MotionIndex,
    groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
    *,
    prediction_fps: float,
    groundtruth_fps: float,
) -> IR2Score:
    """Return the binary macro-average Speed instruction response rate."""
    values: list[float] = []
    sample_ids: list[str] = []
    invalid: dict[str, str] = {}
    directions = {"slow": "prediction_duration > groundtruth_duration", "fast": "prediction_duration < groundtruth_duration"}
    for sample_id in sorted(set(prediction.samples) & set(groundtruth.samples)):
        try:
            metadata = task_metadata_by_id.get(sample_id, {})
            if str(metadata.get("task_family", "")).casefold() != "speed":
                raise ValueError("IR_2 Speed requires Level2 metadata.task_family == 'Speed'")
            direction = str(metadata.get("task_type", "")).casefold()
            if direction not in directions:
                raise ValueError("metadata.task_type must be 'slow' or 'fast'")
            pred_duration, _ = _duration(prediction.samples[sample_id], prediction_fps)
            gt_duration, _ = _duration(groundtruth.samples[sample_id], groundtruth_fps)
            success = pred_duration > gt_duration if direction == "slow" else pred_duration < gt_duration
            sample_ids.append(sample_id)
            values.append(float(success))
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not values:
        raise RuntimeError("no valid prediction/GT pairs for Level-2 Speed IR_2")
    array = np.asarray(values, dtype=np.float64)
    return IR2Score(
        value=float(array.mean()),
        sample_ids=sample_ids,
        sample_values=array,
        details={
            "definition": "binary_duration_order_macro_mean",
            "task_family": "Speed",
            "aggregation": "macro_mean_over_valid_prediction_gt_pairs",
            "slow_rule": directions["slow"],
            "fast_rule": directions["fast"],
            "duration": "(min(common body_pos/body_quat/joint_pos CSV frames) - 1) / side_fps",
            "prediction_fps": prediction_fps,
            "motion_groundtruth_fps": groundtruth_fps,
            "num_valid_samples": len(sample_ids),
            "num_invalid_samples": len(invalid),
            "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        },
    )


def amplitude_ir_2(
    prediction: MotionIndex,
    groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
) -> IR2Score:
    """Return the binary macro-average amplitude instruction response rate."""
    values: list[float] = []
    sample_ids: list[str] = []
    invalid: dict[str, str] = {}
    directions = {
        "scale_up": "prediction_max_joint_angle_range > groundtruth_max_joint_angle_range",
        "scale_down": "prediction_max_joint_angle_range < groundtruth_max_joint_angle_range",
    }
    for sample_id in sorted(set(prediction.samples) & set(groundtruth.samples)):
        try:
            metadata = task_metadata_by_id.get(sample_id, {})
            if str(metadata.get("task_family", "")).casefold() != "amplitude":
                raise ValueError("IR_2 Amplitude requires Level2 metadata.task_family == 'Amplitude'")
            task_type = str(metadata.get("task_type", "")).casefold()
            if task_type not in directions:
                raise ValueError("metadata.task_type must be 'scale_up' or 'scale_down'")
            prediction_amplitude = _maximum_joint_angle_range(prediction.samples[sample_id])
            groundtruth_amplitude = _maximum_joint_angle_range(groundtruth.samples[sample_id])
            success = (prediction_amplitude > groundtruth_amplitude if task_type == "scale_up"
                       else prediction_amplitude < groundtruth_amplitude)
            sample_ids.append(sample_id)
            values.append(float(success))
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not values:
        raise RuntimeError("no valid prediction/GT pairs for Level-2 Amplitude IR_2")
    array = np.asarray(values, dtype=np.float64)
    return IR2Score(
        value=float(array.mean()),
        sample_ids=sample_ids,
        sample_values=array,
        details={
            "definition": "binary_global_maximum_joint_angle_range_comparison",
            "task_family": "Amplitude",
            "aggregation": "macro_mean_over_valid_prediction_gt_pairs",
            "amplitude": "max_joint_j(max_time_t(angle[t,j]) - min_time_t(angle[t,j]))",
            "angle_unit": "radians (joint_pos.csv native unit)",
            "scale_up_rule": directions["scale_up"],
            "scale_down_rule": directions["scale_down"],
            "equal_amplitudes": "failure for both scale_up and scale_down",
            "num_valid_samples": len(sample_ids),
            "num_invalid_samples": len(invalid),
            "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        },
    )


def _maximum_joint_angle_range(clip: Path) -> float:
    values = np.loadtxt(clip / "joint_pos.csv", delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 29 or values.shape[0] < 2:
        raise ValueError(f"{clip}/joint_pos.csv: expected at least two frames of 29 joint angles")
    if not np.isfinite(values).all():
        raise ValueError(f"{clip}/joint_pos.csv: non-finite joint angle")
    return float(np.max(np.max(values, axis=0) - np.min(values, axis=0)))


DIRECTION_MIN_DISPLACEMENT_METERS = 0.10
DIRECTION_LATERAL_TOLERANCE_DEGREES = 60.0


def direction_ir_2(
    prediction: MotionIndex,
    groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
) -> IR2Score:
    """Return binary compliance for a root trajectory moving left or right."""
    values: list[float] = []
    sample_ids: list[str] = []
    invalid: dict[str, str] = {}
    displacement_values: list[float] = []
    angle_values: list[float] = []
    direction_satisfied: list[float] = []
    directions = {"left": math.pi / 2.0, "right": -math.pi / 2.0}
    tolerance = math.radians(DIRECTION_LATERAL_TOLERANCE_DEGREES)

    for sample_id in sorted(set(prediction.samples) & set(groundtruth.samples)):
        try:
            metadata = task_metadata_by_id.get(sample_id, {})
            family = "".join(
                character for character in str(metadata.get("task_family", "")).casefold()
                if character.isalnum()
            )
            if family != "direction":
                raise ValueError("IR_2 Direction requires metadata.task_family == 'Direction'")
            target = str(metadata.get("task_type", "")).casefold()
            if target not in directions:
                raise ValueError("metadata.task_type must be 'left' or 'right'")

            displacement, angle = _initial_heading_relative_displacement(
                load_qpos_36(prediction.samples[sample_id])
            )
            angular_error = abs(_wrap_angle(angle - directions[target]))
            moved = displacement >= DIRECTION_MIN_DISPLACEMENT_METERS
            correct_direction = angular_error <= tolerance
            sample_ids.append(sample_id)
            displacement_values.append(displacement)
            angle_values.append(math.degrees(angle))
            direction_satisfied.append(float(correct_direction))
            values.append(float(moved and correct_direction))
        except Exception as exc:
            invalid[sample_id] = str(exc)

    if not values:
        raise RuntimeError("no valid prediction/GT pairs for Level-2 Direction IR_2")
    array = np.asarray(values, dtype=np.float64)
    return IR2Score(
        value=float(array.mean()),
        sample_ids=sample_ids,
        sample_values=array,
        details={
            "definition": "binary_initial_heading_relative_lateral_displacement_sector",
            "task_family": "Direction",
            "aggregation": "macro_mean_over_valid_prediction_gt_pairs",
            "heading_reference": "initial root quaternion; local +X is forward and local +Y is left",
            "displacement": "prediction_root_xy_last - prediction_root_xy_first",
            "signed_angle": "atan2(cross(initial_forward_xy, displacement_xy), dot(initial_forward_xy, displacement_xy))",
            "left_rule": "displacement_m >= 0.10 and abs(signed_angle_deg - 90) <= 60",
            "right_rule": "displacement_m >= 0.10 and abs(signed_angle_deg + 90) <= 60",
            "minimum_displacement_m": DIRECTION_MIN_DISPLACEMENT_METERS,
            "lateral_sector_center_degrees": {"left": 90.0, "right": -90.0},
            "lateral_sector_tolerance_degrees": DIRECTION_LATERAL_TOLERANCE_DEGREES,
            "mean_horizontal_displacement_m": float(np.mean(displacement_values)),
            "mean_signed_angle_degrees": float(np.mean(angle_values)),
            "num_direction_sector_satisfied": int(sum(direction_satisfied)),
            "num_valid_samples": len(sample_ids),
            "num_invalid_samples": len(invalid),
            "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        },
    )


def _initial_heading_relative_displacement(qpos: np.ndarray) -> tuple[float, float]:
    """Return horizontal displacement magnitude and signed initial-heading-relative angle."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 7:
        raise ValueError("expected at least two qpos frames containing root xyz and quaternion wxyz")
    displacement = values[-1, :2] - values[0, :2]
    distance = float(np.linalg.norm(displacement))
    if not np.isfinite(distance):
        raise ValueError("non-finite root displacement")
    rotation = _root_rotation_matrices(values[:1, 3:7])[0]
    forward = rotation[:2, 0]
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm < 1e-8:
        raise ValueError("initial root forward axis has no horizontal component")
    if distance < 1e-8:
        return distance, 0.0
    forward /= forward_norm
    direction = displacement / distance
    angle = math.atan2(
        float(forward[0] * direction[1] - forward[1] * direction[0]),
        float(np.dot(forward, direction)),
    )
    return distance, angle


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


BODY_RESTRAIN_STATIC_THRESHOLD_MPS = 0.10
BODY_RESTRAIN_MOVE_THRESHOLD_MPS = 0.12
BODY_RESTRAIN_BODY_GROUPS = {
    "arms": (tuple(range(16, 30)), tuple(range(1, 16))),
    "legs": (tuple(range(1, 13)), tuple(range(13, 30))),
}


def body_restrain_ir_2(
    prediction: MotionIndex,
    groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
    *,
    device: str,
) -> IR2Score:
    """Evaluate locked arms/legs and remaining-body motion with local FK speed."""
    evaluator = GlobalMPJPEEvaluator(device=device)
    values: list[float] = []
    sample_ids: list[str] = []
    invalid: dict[str, str] = {}
    lock_values: list[float] = []
    move_values: list[float] = []
    for sample_id in sorted(set(prediction.samples) & set(groundtruth.samples)):
        try:
            metadata = task_metadata_by_id.get(sample_id, {})
            if "".join(character for character in str(metadata.get("task_family", "")).casefold() if character.isalnum()) != "bodyrestrain":
                raise ValueError("IR_2 BodyRestrain requires metadata.task_family == 'BodyRestrain'")
            task_type = str(metadata.get("task_type", "")).casefold()
            if task_type not in BODY_RESTRAIN_BODY_GROUPS:
                raise ValueError("metadata.task_type must be 'arms' or 'legs'")
            restricted, remaining = BODY_RESTRAIN_BODY_GROUPS[task_type]
            velocity = _root_aligned_link_velocity(prediction.samples[sample_id], evaluator)
            restricted_activity = _velocity_rms(velocity, restricted)
            remaining_activity = _velocity_rms(velocity, remaining)
            lock = restricted_activity <= BODY_RESTRAIN_STATIC_THRESHOLD_MPS
            move = remaining_activity >= BODY_RESTRAIN_MOVE_THRESHOLD_MPS
            sample_ids.append(sample_id)
            lock_values.append(float(lock))
            move_values.append(float(move))
            values.append(float(lock and move))
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not values:
        raise RuntimeError("no valid prediction/GT pairs for Level-2 BodyRestrain IR_2")
    array = np.asarray(values, dtype=np.float64)
    return IR2Score(
        value=float(array.mean()),
        sample_ids=sample_ids,
        sample_values=array,
        details={
            "definition": "binary_root_aligned_local_link_velocity_rms",
            "task_family": "BodyRestrain",
            "aggregation": "macro_mean_over_valid_prediction_gt_pairs",
            "restricted_condition": "restricted_velocity_rms_mps <= 0.10",
            "remaining_condition": "remaining_velocity_rms_mps >= 0.12",
            "static_velocity_threshold_mps": BODY_RESTRAIN_STATIC_THRESHOLD_MPS,
            "move_velocity_threshold_mps": BODY_RESTRAIN_MOVE_THRESHOLD_MPS,
            "velocity": "root_aligned_G1_FK_link_position_delta_per_1_over_50_second_RMS",
            "arms_restricted_body_indices": list(BODY_RESTRAIN_BODY_GROUPS["arms"][0]),
            "arms_remaining_body_indices": list(BODY_RESTRAIN_BODY_GROUPS["arms"][1]),
            "legs_restricted_body_indices": list(BODY_RESTRAIN_BODY_GROUPS["legs"][0]),
            "legs_remaining_body_indices": list(BODY_RESTRAIN_BODY_GROUPS["legs"][1]),
            "num_lock_satisfied": int(sum(lock_values)),
            "num_move_satisfied": int(sum(move_values)),
            "num_valid_samples": len(sample_ids),
            "num_invalid_samples": len(invalid),
            "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        },
    )


def _root_aligned_link_velocity(clip: Path, evaluator: GlobalMPJPEEvaluator) -> np.ndarray:
    qpos = load_qpos_36(clip)
    with evaluator.torch.inference_mode():
        positions = evaluator.kinematics.forward_kinematics(
            evaluator.torch.as_tensor(qpos[None], dtype=evaluator.torch.float32, device=evaluator.device)
        )["body_pos_w"][0].detach().cpu().numpy()
    local = np.einsum(
        "tji,tik->tjk",
        positions - positions[:, :1],
        _root_rotation_matrices(qpos[:, 3:7]),
    )
    return np.diff(local, axis=0) * 50.0


def _root_rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack((
        np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)), axis=-1),
        np.stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)), axis=-1),
        np.stack((2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), axis=-1),
    ), axis=1)


def _velocity_rms(velocity: np.ndarray, body_indices: tuple[int, ...]) -> float:
    values = np.asarray(velocity[:, body_indices], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=-1))))


def _duration(clip: Path, fps: float) -> tuple[float, int]:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"FPS must be positive and finite, got {fps!r}")
    frames = min(_csv_data_rows(clip / filename) for filename in ("body_pos.csv", "body_quat.csv", "joint_pos.csv"))
    if frames < 2:
        raise ValueError(f"{clip}: fewer than two common CSV frames")
    return (frames - 1) / fps, frames


def _csv_data_rows(path: Path) -> int:
    content = path.read_bytes()
    newline = content.find(b"\n")
    if newline < 0:
        raise ValueError(f"{path}: missing CSV header")
    rows = content[newline + 1:].rstrip(b"\x00\r\n \t")
    return 0 if not rows else rows.count(b"\n") + 1
