"""Canonical mapping from public Level-1 task roots to RoboSteer task types."""

from __future__ import annotations

from pathlib import Path


_PATH_TO_TASK_TYPE = (
    ("text_to_motion_generation", "motion_generation"),
    ("audio_to_motion_generation", "motion_generation"),
    ("video_to_motion_imitation", "motion_generation"),
    ("rhythm_to_motion_alignment", "motion_generation"),
    ("rotation_to_pose_generation", "motion_generation"),
    ("motion_prediction", "motion_fore"),
    ("motion_retrodiction", "motion_retro"),
    ("motion_interpolation", "motion_inter"),
    ("key_frame_conditioning", "keyframe_conditioning"),
    ("target_reaching", "target_reaching"),
    ("upper_to_full_body_completion", "upper_full"),
    ("lower_to_full_body_completion", "lower_full"),
)


def infer_task_type(task_root: Path) -> str:
    """Infer one of the eight Level-1 IR₁ task types from a public task path."""
    normalized = str(task_root.resolve()).replace("\\", "/").casefold()
    matches = [task_type for marker, task_type in _PATH_TO_TASK_TYPE if marker in normalized]
    if len(set(matches)) != 1:
        raise ValueError(
            f"cannot infer one Level-1 task type from {task_root}; "
            "pass a task root below a known public Level-1 task family"
        )
    return matches[0]
