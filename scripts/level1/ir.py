"""Level-1 intention-realization dispatch without directory-name inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.evaluation.core_metric.BehaviorSteer_level1 import (
    KEYFRAME_TASK_TYPE,
    SPATIAL_COMPLETION_JOINT_GROUPS,
    TARGET_REACHING_TASK_TYPE,
    keyframe_ir_1,
    spatial_joint_range_overlap_ir_1,
    target_reaching_ir_1,
    temporal_iou_ir_1,
)


def compute_ir1(
    task_type: str,
    run_dir: Path,
    *,
    keyframe_task_root: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute IR₁ for one explicitly selected Level-1 task type."""
    if task_type == "motion_generation":
        return 1.0, {"definition": "complete_conditioning_generation_constant", "value": 1.0}
    if task_type in {"motion_fore", "motion_retro", "motion_inter"}:
        return temporal_iou_ir_1(run_dir, workers=32)
    if task_type == KEYFRAME_TASK_TYPE:
        if keyframe_task_root is None:
            raise ValueError("keyframe_conditioning requires --keyframe-task-root")
        return keyframe_ir_1(
            run_dir,
            workers=32,
            fk_batch_size=4096,
            tau_meters=0.20,
            task_root=keyframe_task_root,
            prediction_window_frames=5,
        )
    if task_type == TARGET_REACHING_TASK_TYPE:
        return target_reaching_ir_1(
            run_dir,
            workers=32,
            fk_batch_size=4096,
            tau_meters=0.20,
            prediction_window_frames=5,
        )
    if task_type in SPATIAL_COMPLETION_JOINT_GROUPS:
        return spatial_joint_range_overlap_ir_1(
            run_dir,
            task_type=task_type,
            workers=32,
            static_range_epsilon_rad=1e-4,
            static_position_tolerance_rad=0.05,
        )
    raise ValueError(f"unsupported Level-1 task_type {task_type!r}")
