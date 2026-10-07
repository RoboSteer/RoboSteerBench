"""Time-domain preprocessing for qpos sequences."""

from __future__ import annotations

import numpy as np


def resample_qpos(
    qpos: np.ndarray,
    source_fps: float,
    target_fps: float,
    *,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Convert sampling rate without changing the selected physical duration.

    Root position and joint angles use linear interpolation. Root rotations use
    shortest-path quaternion SLERP and are normalized afterwards. When the
    task's authoritative duration is supplied, source frames are placed
    uniformly on ``[0, duration]``. Otherwise the legacy observed duration
    ``(T - 1) / source_fps`` is used.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(f"qpos must have shape (T, 36), T >= 2; got {qpos.shape}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if duration_seconds is None:
        duration = (len(qpos) - 1) / float(source_fps)
    else:
        duration = float(duration_seconds)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_seconds must be a positive finite number")
    source_time = np.linspace(0.0, duration, len(qpos), dtype=np.float64)
    target_time = np.arange(
        int(np.floor(duration * float(target_fps) + 1e-6)) + 1,
        dtype=np.float64,
    ) / float(target_fps)
    # Avoid interpolation only when the supplied time axis is exactly the
    # conventional observed-FPS one. A task duration can differ slightly even
    # if source_fps equals target_fps.
    observed_source_time = np.arange(len(qpos), dtype=np.float64) / float(source_fps)
    if len(target_time) == len(qpos) and np.allclose(
        source_time, observed_source_time, rtol=0.0, atol=1e-9
    ):
        return qpos
    result = np.empty((len(target_time), 36), dtype=np.float32)
    non_rotation = list(range(3)) + list(range(7, 36))
    for column in non_rotation:
        result[:, column] = np.interp(target_time, source_time, qpos[:, column])
    result[:, 3:7] = _slerp_times(source_time, qpos[:, 3:7], target_time)
    return result


def resample_qpos_to_frame_count(qpos: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample one qpos sequence to an exact frame count.

    This endpoint-preserving temporal normalization is intended for metrics
    that require one-to-one frame correspondence. Root translation and joints
    use linear interpolation; root rotation uses shortest-path SLERP. It does
    not modify the source CSV or imply a change to the source sequence's FPS.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(f"qpos must have shape (T, 36), T >= 2; got {qpos.shape}")
    if isinstance(target_frames, bool) or int(target_frames) != target_frames:
        raise ValueError(f"target_frames must be an integer, got {target_frames!r}")
    target_frames = int(target_frames)
    if target_frames < 2:
        raise ValueError(f"target_frames must be at least 2, got {target_frames}")
    if len(qpos) == target_frames:
        return qpos
    source_time = np.linspace(0.0, 1.0, len(qpos), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    result = np.empty((target_frames, 36), dtype=np.float32)
    non_rotation = list(range(3)) + list(range(7, 36))
    for column in non_rotation:
        result[:, column] = np.interp(target_time, source_time, qpos[:, column])
    result[:, 3:7] = _slerp_times(source_time, qpos[:, 3:7], target_time)
    return result


def _slerp_times(source_time: np.ndarray, quaternion: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_time, target_time, side="right")
    right = np.clip(right, 1, len(source_time) - 1)
    left = right - 1
    denominator = source_time[right] - source_time[left]
    weight = np.divide(
        target_time - source_time[left],
        denominator,
        out=np.zeros_like(target_time),
        where=denominator > 0,
    )[:, None]
    q0, q1 = quaternion[left].astype(np.float64), quaternion[right].astype(np.float64)
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    linear = (1.0 - weight) * q0 + weight * q1
    spherical = (
        np.sin((1.0 - weight) * angle) / np.maximum(sin_angle, 1e-8) * q0
        + np.sin(weight * angle) / np.maximum(sin_angle, 1e-8) * q1
    )
    output = np.where(sin_angle < 1e-6, linear, spherical)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-8)
    return output.astype(np.float32)
