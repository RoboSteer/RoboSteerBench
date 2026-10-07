"""Prepare fixed-length uniformly sampled windows from resampled motions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import numpy as np

from scripts.evaluation.data.motion import MotionIndex
from scripts.evaluation.runtime.motion import ResampledMotionIndex, load_resampled_motion_index


WINDOW_ID_SEPARATOR = "::window_"


@dataclass(frozen=True)
class PreparedMotion:
    """MotionEncoder inputs, with one row per fixed temporal window."""

    sample_ids: tuple[str, ...]
    source_sample_ids: tuple[str, ...]
    window_indices: np.ndarray
    window_starts: np.ndarray
    qpos_36: np.ndarray
    invalid: dict[str, str]


def prepare_motion_index(
    index: MotionIndex,
    *,
    source_fps: float,
    target_fps: float,
    window_frames: int,
    num_windows: int,
    include_sample_ids: Collection[str] | None = None,
    resampled: ResampledMotionIndex | None = None,
    workers: int = 0,
) -> PreparedMotion:
    """Draw fixed windows from reusable complete resampled motions.

    When ``resampled`` is supplied, no CSV is reread and no resampling is
    repeated. This lets FID/Diversity and complete-motion metrics share the
    same CPU preparation within one evaluation run.
    """
    if window_frames <= 1:
        raise ValueError("window_frames must be greater than one")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if resampled is None:
        resampled = load_resampled_motion_index(
            index,
            source_fps=source_fps,
            target_fps=target_fps,
            include_sample_ids=include_sample_ids,
            workers=workers,
        )
    if abs(resampled.source_fps - float(source_fps)) > 1e-6 or abs(resampled.target_fps - float(target_fps)) > 1e-6:
        raise ValueError("provided resampled motions use a different FPS protocol")
    selected = None if include_sample_ids is None else set(include_sample_ids)
    source_ids = [sample_id for sample_id in resampled.sample_ids if selected is None or sample_id in selected]
    sample_ids: list[str] = []
    source_sample_ids: list[str] = []
    window_indices: list[int] = []
    window_starts: list[int] = []
    sequences: list[np.ndarray] = []
    invalid = dict(resampled.invalid)
    for source_id in source_ids:
        qpos = resampled.qpos_by_id[source_id]
        try:
            if len(qpos) < window_frames:
                raise ValueError(f"only {len(qpos)} frames after resampling; require {window_frames}")
            for window_index, start in enumerate(uniform_window_starts(len(qpos), window_frames, num_windows)):
                sample_ids.append(window_sample_id(source_id, window_index))
                source_sample_ids.append(source_id)
                window_indices.append(window_index)
                window_starts.append(start)
                sequences.append(qpos[start : start + window_frames])
        except Exception as exc:
            invalid[source_id] = str(exc)
    if not sequences:
        raise RuntimeError(f"no valid motion samples under {index.root}")
    return PreparedMotion(
        sample_ids=tuple(sample_ids),
        source_sample_ids=tuple(source_sample_ids),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        window_starts=np.asarray(window_starts, dtype=np.int32),
        qpos_36=np.stack(sequences).astype(np.float32),
        invalid=invalid,
    )


def uniform_window_starts(length: int, window_frames: int, num_windows: int) -> np.ndarray:
    """Return deterministic window starts from the beginning through the end."""
    if length < window_frames:
        raise ValueError("motion is shorter than the requested window")
    if num_windows == 1:
        return np.zeros(1, dtype=np.int32)
    return np.rint(np.linspace(0, length - window_frames, num=num_windows)).astype(np.int32)


def window_sample_id(source_sample_id: str, window_index: int) -> str:
    return f"{source_sample_id}{WINDOW_ID_SEPARATOR}{window_index:02d}"


def source_sample_id(window_id: str) -> str:
    """Recover the source sample id from a prepared window id."""
    source, separator, suffix = window_id.rpartition(WINDOW_ID_SEPARATOR)
    if not separator or not suffix.isdigit():
        raise ValueError(f"invalid window sample id {window_id!r}")
    return source
