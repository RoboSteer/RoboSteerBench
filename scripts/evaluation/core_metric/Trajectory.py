"""Pure-motion heuristic compliance for Level-2 trajectory tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.motion import MotionIndex, load_qpos_36


TASK_TYPES = frozenset(("back_and_forth", "clockwise_or_counterclockwise", "s_shape"))
MIN_DISPLACEMENT_METERS = 0.25
EFFECTIVE_STEP_METERS = 0.005
BACK_FORTH_END_RATIO_MAX = 0.25
BACK_FORTH_PATH_RATIO_MIN = 1.60
BACK_FORTH_DIRECTION_RATIO_MIN = 0.30
BACK_FORTH_AREA_RATIO_MAX = 0.05
TURN_AREA_RATIO_MIN = 0.10
TURN_ANGLE_MIN_RADIANS = math.pi
S_LATERAL_ANGLE_MIN_DEGREES = 15.0
S_LATERAL_AMPLITUDE_RATIO_MIN = 0.15
S_END_RATIO_MIN = 0.35


@dataclass(frozen=True)
class TrajectoryIR2Score:
    value: float
    sample_ids: list[str]
    sample_values: np.ndarray
    details: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryFeatures:
    path_length: float
    max_displacement: float
    start_end_distance: float
    signed_area: float
    normalized_area: float
    total_absolute_turning: float
    principal_positive_distance: float
    principal_negative_distance: float
    s_lateral_pattern: str
    s_lateral_amplitude: float


def trajectory_ir_2(
    prediction: MotionIndex,
    groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
) -> TrajectoryIR2Score:
    """Compute binary compliance from generated root XY trajectories only.

    The Level-2 JSON task type selects the expected family.
    Ground-truth motion and videos are deliberately not used for the decision.
    """
    values: list[float] = []
    sample_ids: list[str] = []
    invalid: dict[str, str] = {}
    feature_rows: list[TrajectoryFeatures] = []
    counts = {kind: 0 for kind in sorted(TASK_TYPES)}
    successes = {kind: 0 for kind in sorted(TASK_TYPES)}
    for sample_id in sorted(set(prediction.samples) & set(groundtruth.samples)):
        try:
            metadata = task_metadata_by_id.get(sample_id, {})
            family = _normalized_name(str(metadata.get("task_family", "")))
            if family != "trajectory":
                raise ValueError("Trajectory IR_2 requires metadata.task_family == 'Trajectory'")
            task_type = str(metadata.get("task_type", "")).casefold()
            if task_type not in TASK_TYPES:
                raise ValueError(f"unsupported trajectory task type {task_type!r}")
            features = trajectory_features(load_qpos_36(prediction.samples[sample_id]))
            passed = trajectory_compliance(task_type, features)
            sample_ids.append(sample_id)
            values.append(float(passed))
            feature_rows.append(features)
            counts[task_type] += 1
            successes[task_type] += int(passed)
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not values:
        raise RuntimeError("no valid prediction/GT pairs for Level-2 Trajectory IR_2")
    result = np.asarray(values, dtype=np.float64)
    return TrajectoryIR2Score(
        value=float(result.mean()),
        sample_ids=sample_ids,
        sample_values=result,
        details={
            "definition": "binary_generated_root_planar_trajectory_heuristic",
            "task_family": "Trajectory",
            "aggregation": "macro_mean_over_valid_prediction_gt_pairs",
            "trajectory_source": "body_pos.csv root (body_0) projected to horizontal XY plane",
            "preprocessing": (
                "five-frame edge-padded moving-average smoothing with preserved endpoints; "
                "remove consecutive points below max(0.005 m, 0.01 * max_displacement)"
            ),
            "back_and_forth_rule": (
                "max displacement >= 0.25 m; return ratio <= 0.25; path ratio >= 1.60; "
                "positive and negative principal-axis travel >= 0.30 each; normalized area <= 0.05"
            ),
            "clockwise_or_counterclockwise_rule": (
                "max displacement >= 0.25 m; normalized absolute area >= 0.10; "
                "cumulative absolute turn >= pi radians; orientation is ignored"
            ),
            "s_shape_rule": (
                "four arc-length-equidistant points yield left-right-left or right-left-right lateral segments; "
                "each segment's lateral angle >= 15 degrees; normalized lateral amplitude >= 0.15; "
                "end ratio >= 0.35"
            ),
            "thresholds": {
                "minimum_displacement_m": MIN_DISPLACEMENT_METERS,
                "effective_step_m": EFFECTIVE_STEP_METERS,
                "back_and_forth_end_ratio_max": BACK_FORTH_END_RATIO_MAX,
                "back_and_forth_path_ratio_min": BACK_FORTH_PATH_RATIO_MIN,
                "back_and_forth_direction_ratio_min": BACK_FORTH_DIRECTION_RATIO_MIN,
                "back_and_forth_area_ratio_max": BACK_FORTH_AREA_RATIO_MAX,
                "turn_area_ratio_min": TURN_AREA_RATIO_MIN,
                "turn_angle_min_radians": TURN_ANGLE_MIN_RADIANS,
                "s_lateral_angle_min_degrees": S_LATERAL_ANGLE_MIN_DEGREES,
                "s_lateral_amplitude_ratio_min": S_LATERAL_AMPLITUDE_RATIO_MIN,
                "s_end_ratio_min": S_END_RATIO_MIN,
            },
            "per_task": {
                kind: {
                    "num_valid_samples": counts[kind],
                    "num_satisfied": successes[kind],
                    "compliance": None if counts[kind] == 0 else successes[kind] / counts[kind],
                }
                for kind in sorted(TASK_TYPES)
            },
            "mean_features": mean_features(feature_rows),
            "num_valid_samples": len(sample_ids),
            "num_invalid_samples": len(invalid),
            "invalid_samples": dict(list(sorted(invalid.items()))[:32]),
        },
    )


def trajectory_type_from_directory(name: str) -> str:
    matches = [kind for kind in TASK_TYPES if name.casefold().endswith("_" + kind)]
    if len(matches) != 1:
        expected = ", ".join("_" + kind for kind in sorted(TASK_TYPES))
        raise ValueError(f"prediction directory must end in one of: {expected}")
    return matches[0]


def trajectory_compliance(task_type: str, features: TrajectoryFeatures) -> bool:
    if features.max_displacement < MIN_DISPLACEMENT_METERS:
        return False
    scale = features.max_displacement
    end_ratio = features.start_end_distance / scale
    if task_type == "back_and_forth":
        return (
            end_ratio <= BACK_FORTH_END_RATIO_MAX
            and features.path_length / scale >= BACK_FORTH_PATH_RATIO_MIN
            and features.principal_positive_distance / scale >= BACK_FORTH_DIRECTION_RATIO_MIN
            and features.principal_negative_distance / scale >= BACK_FORTH_DIRECTION_RATIO_MIN
            and features.normalized_area <= BACK_FORTH_AREA_RATIO_MAX
        )
    if task_type == "clockwise_or_counterclockwise":
        return (
            features.normalized_area >= TURN_AREA_RATIO_MIN
            and features.total_absolute_turning >= TURN_ANGLE_MIN_RADIANS
        )
    if task_type == "s_shape":
        return (
            features.s_lateral_pattern in {"left-right-left", "right-left-right"}
            and features.s_lateral_amplitude / scale >= S_LATERAL_AMPLITUDE_RATIO_MIN
            and end_ratio >= S_END_RATIO_MIN
        )
    raise ValueError(f"unsupported trajectory task type {task_type!r}")


def trajectory_features(qpos: np.ndarray) -> TrajectoryFeatures:
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("expected at least two qpos frames containing root XY position")
    points = values[:, :2]
    if not np.isfinite(points).all():
        raise ValueError("non-finite root XY position")
    points = smooth_path(points)
    prefilter_maximum = float(np.max(np.linalg.norm(points - points[0], axis=1)))
    points = effective_path(points, prefilter_maximum)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    path_length = float(lengths.sum())
    maximum = float(np.max(np.linalg.norm(points - points[0], axis=1)))
    start_end = float(np.linalg.norm(points[-1] - points[0]))
    signed_area = polygon_area(points)
    normalized_area = 0.0 if maximum < 1e-8 else abs(signed_area) / maximum**2
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    total_turn = float(np.abs(wrap_angle(np.diff(headings))).sum()) if len(headings) > 1 else 0.0
    principal = principal_axis(points)
    longitudinal = vectors @ principal
    positive = float(longitudinal[longitudinal > 0.0].sum())
    negative = float(-longitudinal[longitudinal < 0.0].sum())
    pattern, amplitude = s_shape_pattern(points, principal)
    return TrajectoryFeatures(
        path_length, maximum, start_end, signed_area, normalized_area, total_turn,
        positive, negative, pattern, amplitude,
    )


def smooth_path(points: np.ndarray) -> np.ndarray:
    if len(points) < 5:
        return points.copy()
    padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
    kernel = np.ones(5, dtype=np.float64) / 5.0
    result = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)
    result[0], result[-1] = points[0], points[-1]
    return result


def effective_path(points: np.ndarray, maximum: float) -> np.ndarray:
    threshold = max(EFFECTIVE_STEP_METERS, 0.01 * maximum)
    result = [points[0]]
    for point in points[1:-1]:
        if float(np.linalg.norm(point - result[-1])) >= threshold:
            result.append(point)
    if float(np.linalg.norm(points[-1] - result[-1])) >= 1e-10:
        result.append(points[-1])
    return np.asarray(result, dtype=np.float64)


def polygon_area(points: np.ndarray) -> float:
    other = np.roll(points, -1, axis=0)
    return float(0.5 * np.sum(points[:, 0] * other[:, 1] - points[:, 1] * other[:, 0]))


def principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.array((1.0, 0.0), dtype=np.float64)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    if float(np.dot(axis, points[-1] - points[0])) < 0.0:
        axis = -axis
    return axis


def s_shape_pattern(points: np.ndarray, principal: np.ndarray) -> tuple[str, float]:
    landmarks = resample_path(points, 4)
    lateral_axis = np.array((-principal[1], principal[0]), dtype=np.float64)
    segments = np.diff(landmarks, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    lateral = segments @ lateral_axis
    minimum = math.sin(math.radians(S_LATERAL_ANGLE_MIN_DEGREES))
    labels: list[str] = []
    for component, length in zip(lateral, lengths):
        if length < 1e-8 or abs(component) / length < minimum:
            return "", float(np.ptp(landmarks @ lateral_axis))
        labels.append("left" if component > 0.0 else "right")
    return "-".join(labels), float(np.ptp(landmarks @ lateral_axis))


def resample_path(points: np.ndarray, count: int) -> np.ndarray:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] < 1e-8:
        return np.repeat(points[:1], count, axis=0)
    output = np.empty((count, 2), dtype=np.float64)
    segment = 0
    for index, target in enumerate(np.linspace(0.0, cumulative[-1], count)):
        while segment < len(lengths) - 1 and cumulative[segment + 1] < target:
            segment += 1
        alpha = 0.0 if lengths[segment] < 1e-8 else (target - cumulative[segment]) / lengths[segment]
        output[index] = points[segment] + alpha * vectors[segment]
    return output


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def mean_features(rows: list[TrajectoryFeatures]) -> dict[str, float]:
    if not rows:
        return {}
    names = (
        "path_length", "max_displacement", "start_end_distance", "signed_area",
        "normalized_area", "total_absolute_turning", "principal_positive_distance",
        "principal_negative_distance", "s_lateral_amplitude",
    )
    return {name: float(np.mean([getattr(row, name) for row in rows])) for name in names}


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
