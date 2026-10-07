"""Task-family dispatch for Level-2 intention realization.

Metric modules are imported only for the selected family so the CLI can parse
arguments without initializing optional FK/trajectory dependencies.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.evaluation.data.motion import MotionIndex


def compute_ir2(
    *,
    family: str,
    prediction: MotionIndex,
    base_groundtruth: MotionIndex,
    task_metadata_by_id: dict[str, dict[str, Any]],
    prediction_fps: float,
    groundtruth_fps: float,
    device: str,
):
    """Compute one supported IR_2 against the original Level-1 GT."""
    normalized = re.sub(r"[^a-z0-9]", "", family.casefold())
    if normalized == "speed":
        from scripts.evaluation.core_metric.BehaviorSteer_level2 import speed_ir_2
        return speed_ir_2(prediction, base_groundtruth, task_metadata_by_id,
                          prediction_fps=prediction_fps, groundtruth_fps=groundtruth_fps)
    if normalized == "amplitude":
        from scripts.evaluation.core_metric.BehaviorSteer_level2 import amplitude_ir_2
        return amplitude_ir_2(prediction, base_groundtruth, task_metadata_by_id)
    if normalized == "bodyrestrain":
        from scripts.evaluation.core_metric.BehaviorSteer_level2 import body_restrain_ir_2
        return body_restrain_ir_2(prediction, base_groundtruth, task_metadata_by_id, device=device)
    if normalized == "direction":
        from scripts.evaluation.core_metric.BehaviorSteer_level2 import direction_ir_2
        return direction_ir_2(prediction, base_groundtruth, task_metadata_by_id)
    if normalized == "trajectory":
        from scripts.evaluation.core_metric.Trajectory import trajectory_ir_2
        return trajectory_ir_2(prediction, base_groundtruth, task_metadata_by_id)
    if normalized in {"order", "times"}:
        raise NotImplementedError(f"IR_2 for Level-2 {family} is not implemented yet")
    raise ValueError(f"unsupported Level-2 task family {family!r}")
