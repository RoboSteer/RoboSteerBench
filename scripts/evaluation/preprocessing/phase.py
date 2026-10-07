"""Duration-invariant phase normalization for evaluator motion inputs.

The TextOp OMG formal protocol represents every complete source motion by one
fixed-length sequence.  Its samples lie uniformly on the motion's own
``0% .. 100%`` phase axis rather than on a shared physical-FPS timeline.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Collection, Iterator, Mapping

import numpy as np
from tqdm.auto import tqdm

from scripts.evaluation.data.motion import MotionIndex, load_qpos_36

from .motion import PreparedMotion


def phase_normalize_qpos(
    qpos_36: np.ndarray, *, target_frames: int = 60, duration_seconds: float | None = None,
) -> np.ndarray:
    """Resample one complete qpos trajectory over its inclusive phase axis.

    This follows the OMG TextOp artifact convention: positions and joints use
    linear interpolation, quaternions use sign-consistent normalized linear
    interpolation (NLERP), and both source endpoints are retained.
    """
    qpos = np.asarray(qpos_36, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"qpos_36 must have shape (T, 36), got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError("phase normalization requires at least two source frames")
    if target_frames < 2:
        raise ValueError("phase normalization requires target_frames >= 2")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos_36 contains non-finite values")

    # The formal OMG protocol writes source and target timestamps on the
    # task-defined [0, duration] interval.  The normalized phase values are
    # numerically equivalent, but retaining duration here makes that protocol
    # explicit and auditable.
    duration = 1.0 if duration_seconds is None else float(duration_seconds)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_seconds must be a positive finite number")
    source_phase = np.linspace(0.0, duration, qpos.shape[0], dtype=np.float64)
    target_phase = np.linspace(0.0, duration, int(target_frames), dtype=np.float64)
    result = np.empty((int(target_frames), 36), dtype=np.float32)
    for column in (*range(3), *range(7, 36)):
        result[:, column] = np.interp(target_phase, source_phase, qpos[:, column]).astype(np.float32)

    quaternion = np.asarray(qpos[:, 3:7], dtype=np.float64).copy()
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-8)
    for frame in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[frame - 1], quaternion[frame])) < 0.0:
            quaternion[frame] *= -1.0
    interpolated = np.stack(
        [np.interp(target_phase, source_phase, quaternion[:, column]) for column in range(4)], axis=1
    )
    norm = np.linalg.norm(interpolated, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("invalid root quaternion after phase normalization")
    interpolated /= norm
    # Match the formal artifact's standardized w-positive output convention.
    interpolated[interpolated[:, 0] < 0.0] *= -1.0
    result[:, 3:7] = interpolated.astype(np.float32)
    return result


def prepare_phase_motion_index(
    index: MotionIndex,
    *,
    target_frames: int = 60,
    include_sample_ids: Collection[str] | None = None,
    workers: int = 0,
    description: str | None = None,
    duration_seconds_by_id: Mapping[str, float] | None = None,
) -> PreparedMotion:
    """Load every selected complete motion and phase-normalize it once."""
    selected = None if include_sample_ids is None else set(include_sample_ids)
    items = [
        (sample_id, str(path), int(target_frames), None if duration_seconds_by_id is None else duration_seconds_by_id.get(sample_id))
        for sample_id, path in sorted(index.samples.items())
        if selected is None or sample_id in selected
    ]
    if not items:
        raise RuntimeError(f"no selected motions under {index.root}")

    rows: list[tuple[str, np.ndarray | None, str | None]]
    worker_count = max(int(workers), 0)
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator: Iterator[tuple[str, np.ndarray | None, str | None]] = executor.map(
                _load_and_phase_normalize,
                items,
                chunksize=_chunksize(len(items), worker_count),
            )
            rows = list(
                tqdm(iterator, total=len(items), desc=description or f"Phase-normalizing {index.root.name}",
                     unit="motion", dynamic_ncols=True)
            )
    else:
        rows = list(
            tqdm((_load_and_phase_normalize(item) for item in items), total=len(items),
                 desc=description or f"Phase-normalizing {index.root.name}", unit="motion", dynamic_ncols=True)
        )

    sample_ids: list[str] = []
    sequences: list[np.ndarray] = []
    invalid: dict[str, str] = {}
    for sample_id, qpos, error in rows:
        if qpos is None or error is not None:
            invalid[sample_id] = error or "unknown phase-normalization failure"
            continue
        sample_ids.append(sample_id)
        sequences.append(qpos)
    if not sequences:
        raise RuntimeError(f"no valid motions under {index.root} after phase normalization")
    count = len(sample_ids)
    return PreparedMotion(
        sample_ids=tuple(sample_ids),
        source_sample_ids=tuple(sample_ids),
        window_indices=np.zeros(count, dtype=np.int32),
        window_starts=np.zeros(count, dtype=np.int32),
        qpos_36=np.stack(sequences).astype(np.float32),
        invalid=invalid,
    )


def _load_and_phase_normalize(
    item: tuple[str, str, int, float | None],
) -> tuple[str, np.ndarray | None, str | None]:
    sample_id, path_string, target_frames, duration_seconds = item
    try:
        if duration_seconds is None:
            raise ValueError("missing task metadata.duration")
        return sample_id, phase_normalize_qpos(
            load_qpos_36(Path(path_string)),
            target_frames=target_frames,
            duration_seconds=duration_seconds,
        ), None
    except Exception as exc:
        return sample_id, None, str(exc)


def _chunksize(num_items: int, workers: int) -> int:
    return max(1, min(64, num_items // max(workers * 8, 1)))
