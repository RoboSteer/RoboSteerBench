"""Audio-motion semantic evaluator built around OMG's frozen motion encoder."""

from .config import ProjectConfig
from .inference import AudioMotionEvaluator

__all__ = ["AudioMotionEvaluator", "ProjectConfig"]
