"""Locate the OMG dependency in a source checkout or a downloaded model bundle."""

import os
from pathlib import Path


def omg_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("ROBOOSTEER_OMG_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    bundle = project_root / "models"
    for name in (
        "motion_encoder", "text_motion_evaluator", "audio_motion_evaluator",
        "rhythm_motion_evaluator", "trajectory_motion_evaluator",
        "video_motion_evaluator_human", "video_motion_evaluator_skel",
    ):
        candidate = bundle / name / "omg"
        if (candidate / "src" / "omg" / "__init__.py").is_file():
            return candidate
    return project_root / "third_party" / "OMG"
