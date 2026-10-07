"""Build a unified manifest across prediction, motion GT, and instruction GT."""

from __future__ import annotations

import json
from pathlib import Path

from .instruction import InstructionIndex
from .motion import MotionIndex
from .timing import TaskTimingIndex


def build_sample_manifest(
    prediction: MotionIndex,
    motion_groundtruth: MotionIndex | None,
    instruction_groundtruth: InstructionIndex | None,
    output: Path,
    *,
    task_timing: TaskTimingIndex | None = None,
) -> list[dict[str, object]]:
    sample_ids = set(prediction.samples)
    if motion_groundtruth is not None:
        sample_ids |= set(motion_groundtruth.samples)
    if instruction_groundtruth is not None:
        sample_ids |= set(instruction_groundtruth.samples)
    rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_ids):
        instruction = None if instruction_groundtruth is None else instruction_groundtruth.samples.get(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "prediction_motion": _path(prediction.samples.get(sample_id)),
                "motion_groundtruth": _path(
                    None if motion_groundtruth is None else motion_groundtruth.samples.get(sample_id)
                ),
                "instruction_groundtruth": _path(None if instruction is None else instruction.path),
                "instruction_encoder": None if instruction_groundtruth is None else instruction_groundtruth.encoder,
                "task_duration_seconds": None if task_timing is None else task_timing.durations_seconds.get(sample_id),
                "task_duration_source": None if task_timing is None else "metadata.duration",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _path(path: Path | None) -> str | None:
    return None if path is None else str(path)
