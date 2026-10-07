"""Multiprocess motion preparation and length-aware padded batching.

All consumers retain ownership of their metric definition.  This module only
loads qpos, resamples on CPU, and presents variable-length sequences as padded
batches plus a validity mask so padding can never affect a metric reduction.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterator

import numpy as np
from tqdm.auto import tqdm

from scripts.evaluation.data.motion import MotionIndex, load_qpos_36


@dataclass(frozen=True)
class ResampledMotionIndex:
    """Complete resampled motions keyed by source sample ID."""

    sample_ids: tuple[str, ...]
    qpos_by_id: dict[str, np.ndarray]
    invalid: dict[str, str]
    source_fps: float
    target_fps: float


@dataclass(frozen=True)
class MotionBatch:
    """Length-aware batch of complete qpos motions.

    ``qpos_36`` repeats the last valid pose after each sequence's true length.
    Consumers must use ``valid`` for any framewise aggregation.
    """

    sample_ids: tuple[str, ...]
    qpos_36: np.ndarray
    valid: np.ndarray
    lengths: np.ndarray


def load_resampled_motion_index(
    index: MotionIndex,
    *,
    source_fps: float,
    target_fps: float,
    include_sample_ids: Collection[str] | None = None,
    workers: int = 0,
    description: str | None = None,
) -> ResampledMotionIndex:
    """Load and resample selected qpos motions, optionally with CPU workers.

    Results remain deterministically ordered by sample ID irrespective of
    worker count. ``workers=0`` keeps the serial behaviour for debugging.
    """
    selected = None if include_sample_ids is None else set(include_sample_ids)
    items = [
        (sample_id, str(path), float(source_fps), float(target_fps))
        for sample_id, path in sorted(index.samples.items())
        if selected is None or sample_id in selected
    ]
    if not items:
        return ResampledMotionIndex((), {}, {}, float(source_fps), float(target_fps))
    worker_count = max(int(workers), 0)
    iterator: Iterator[tuple[str, np.ndarray | None, str | None]]
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(_load_and_resample, items, chunksize=_chunksize(len(items), worker_count))
            rows = list(tqdm(iterator, total=len(items), desc=description or f"Preparing {index.root.name}",
                             unit="motion", dynamic_ncols=True))
    else:
        rows = list(tqdm((_load_and_resample(item) for item in items), total=len(items),
                         desc=description or f"Preparing {index.root.name}", unit="motion", dynamic_ncols=True))
    qpos_by_id: dict[str, np.ndarray] = {}
    invalid: dict[str, str] = {}
    for sample_id, qpos, error in rows:
        if error is not None or qpos is None:
            invalid[sample_id] = error or "unknown qpos preprocessing failure"
        else:
            qpos_by_id[sample_id] = qpos
    return ResampledMotionIndex(
        tuple(qpos_by_id), qpos_by_id, invalid, float(source_fps), float(target_fps)
    )


def batch_motion_sequences(
    qpos_by_id: dict[str, np.ndarray],
    *,
    max_frames_per_batch: int,
) -> Iterator[MotionBatch]:
    """Yield deterministically length-bucketed padded batches.

    The frame budget constrains ``batch_size * longest_length``.  Sorting by
    length keeps padding small while preserving deterministic ordering within
    ties.  This policy is metric-independent and suitable for FK or encoders.
    """
    frame_budget = int(max_frames_per_batch)
    if frame_budget <= 0:
        raise ValueError("max_frames_per_batch must be positive")
    ordered = sorted(qpos_by_id.items(), key=lambda item: (len(item[1]), item[0]))
    current: list[tuple[str, np.ndarray]] = []
    current_max = 0
    for sample_id, qpos in ordered:
        _validate_qpos(qpos, sample_id)
        proposed_max = max(current_max, len(qpos))
        if current and (len(current) + 1) * proposed_max > frame_budget:
            yield _pad_batch(current)
            current = []
            current_max = 0
        current.append((sample_id, qpos))
        current_max = max(current_max, len(qpos))
    if current:
        yield _pad_batch(current)


def _load_and_resample(item: tuple[str, str, float, float]) -> tuple[str, np.ndarray | None, str | None]:
    sample_id, path_string, source_fps, target_fps = item
    try:
        qpos = __import__("preprocessing.temporal", fromlist=["resample_qpos"]).resample_qpos(load_qpos_36(Path(path_string)), source_fps, target_fps)
        _validate_qpos(qpos, sample_id)
        return sample_id, qpos, None
    except Exception as exc:
        return sample_id, None, str(exc)


def _pad_batch(items: list[tuple[str, np.ndarray]]) -> MotionBatch:
    sample_ids = tuple(sample_id for sample_id, _ in items)
    lengths = np.asarray([len(qpos) for _, qpos in items], dtype=np.int32)
    max_length = int(lengths.max())
    qpos = np.zeros((len(items), max_length, 36), dtype=np.float32)
    valid = np.zeros((len(items), max_length), dtype=bool)
    for row, (_, sequence) in enumerate(items):
        qpos[row, :len(sequence)] = sequence
        qpos[row, len(sequence):] = sequence[-1]
        valid[row, :len(sequence)] = True
    return MotionBatch(sample_ids, qpos, valid, lengths)


def _validate_qpos(qpos: np.ndarray, sample_id: str) -> None:
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(f"{sample_id}: expected qpos shape (T, 36), T >= 2; got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{sample_id}: qpos contains non-finite values")


def _chunksize(num_items: int, workers: int) -> int:
    return max(1, min(64, num_items // max(workers * 8, 1)))
