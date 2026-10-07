"""Authoritative task-duration metadata and evaluation provenance.

OMG task JSON files define ``metadata.duration``.  Motion CSV files have no
timestamps, so their frame count alone must never be presented as the task's
ground-truth duration.  This module accepts either the compact local JSONL
mirror or an OMG task-JSON directory and exposes a single validated mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Collection, Iterator

import numpy as np

from .motion import normalize_sample_id


@dataclass(frozen=True)
class TaskTimingIndex:
    """Validated mapping from normalized motion ID to task duration in seconds."""

    source: Path
    durations_seconds: dict[str, float]
    record_count: int
    duration_manifest_sha256: str
    source_format: str

    def require(self, sample_ids: Collection[str], *, label: str) -> dict[str, float]:
        """Return requested durations, failing rather than silently filtering.

        OMG's formal protocol treats a missing task duration as an invalid
        provenance record.  Failing here prevents a partial, non-comparable
        FID score from being emitted.
        """
        selected = tuple(sorted(set(sample_ids)))
        missing = [sample_id for sample_id in selected if sample_id not in self.durations_seconds]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise ValueError(
                f"{label}: missing task metadata.duration for {preview}{suffix}; "
                f"metadata source is {self.source}"
            )
        return {sample_id: self.durations_seconds[sample_id] for sample_id in selected}

    def report(self, sample_ids: Collection[str] | None = None) -> dict[str, Any]:
        selected = set(self.durations_seconds) if sample_ids is None else set(sample_ids)
        matched = selected & set(self.durations_seconds)
        missing = sorted(selected - set(self.durations_seconds))
        return {
            "source": str(self.source),
            "source_format": self.source_format,
            "record_count": self.record_count,
            "num_unique_samples": len(self.durations_seconds),
            "duration_field": "metadata.duration",
            "duration_unit": "seconds",
            "duration_manifest_sha256": self.duration_manifest_sha256,
            "num_selected_samples": len(selected),
            "num_selected_with_task_duration": len(matched),
            "num_selected_missing_task_duration": len(missing),
            "missing_sample_ids": missing[:32],
        }


def load_task_timing_index(path: Path) -> TaskTimingIndex:
    """Load a JSONL duration manifest, a task JSON, or a task-JSON directory."""
    source = path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"task metadata source does not exist: {source}")
    if source.is_dir():
        rows = _iter_json_directory(source)
        source_format = "task_json_directory"
    elif source.suffix.lower() == ".jsonl":
        rows = _iter_jsonl(source)
        source_format = "jsonl_duration_manifest"
    elif source.suffix.lower() == ".json":
        rows = _iter_json(source)
        source_format = "task_json"
    else:
        raise ValueError("--task-metadata must be a .jsonl/.json file or a directory of task JSON files")

    durations: dict[str, float] = {}
    record_count = 0
    for origin, payload in rows:
        record_count += 1
        raw_id, duration = _extract_id_and_duration(payload, origin)
        sample_id = _normalize_task_sample_id(raw_id)
        if not sample_id:
            raise ValueError(f"{origin}: empty normalized task sample ID")
        previous = durations.get(sample_id)
        if previous is not None and not np.isclose(previous, duration, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"conflicting metadata.duration for {sample_id}: {previous} vs {duration} ({origin})"
            )
        durations[sample_id] = duration
    if not durations:
        raise RuntimeError(f"no task metadata.duration records found under {source}")
    digest = hashlib.sha256()
    for sample_id, duration in sorted(durations.items()):
        digest.update(f"{sample_id}\t{duration:.9f}\n".encode("utf-8"))
    return TaskTimingIndex(source, durations, record_count, digest.hexdigest(), source_format)


def _normalize_task_sample_id(raw_id: str) -> str:
    """Recover a raw OMG motion ID from a task ID before rollout normalization.

    All current OMG task IDs end in ``<youtube_id>_<clip>_<start>_<end>``,
    where ``youtube_id`` has eleven URL-safe characters.  Matching that suffix
    makes this work for future task families too, rather than hard-coding only
    Text/Rhythm/Trajectory prefixes.
    """
    suffix = re.match(r"^L\d+_.+?([-_A-Za-z0-9]{11}_\d{5}_\d+_\d+)$", raw_id)
    if suffix is not None:
        raw_id = suffix.group(1)
    else:
        raw_id = re.sub(
            r"^L\d+_(?:TXT_(?:GEN|COMP_HANDS|COMP_LEGS|FORE|INTER|RETRO)|MUL_(?:RHY|POS)|INTERLEAVE)_",
            "",
            raw_id,
        )
    return normalize_sample_id(raw_id)


def _iter_jsonl(path: Path) -> Iterator[tuple[Path, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                try:
                    yield path.with_name(f"{path.name}:{line_number}"), json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc


def _iter_json(path: Path) -> Iterator[tuple[Path, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            yield path, row
    else:
        yield path, payload


def _iter_json_directory(root: Path) -> Iterator[tuple[Path, Any]]:
    files = sorted(root.rglob("*.json"))
    if not files:
        raise RuntimeError(f"no .json task files under {root}")
    for path in files:
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON task file {path}") from exc


def _extract_id_and_duration(payload: Any, origin: Path) -> tuple[str, float]:
    if not isinstance(payload, dict):
        raise ValueError(f"{origin}: task metadata row must be a JSON object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    raw_id = metadata.get("task_id") or payload.get("task_id") or payload.get("name") or payload.get("sample_id")
    value = metadata.get("duration", payload.get("duration"))
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError(f"{origin}: missing metadata.task_id or name")
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{origin}: missing or invalid metadata.duration") from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{origin}: metadata.duration must be a positive finite number")
    return raw_id, duration
