from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .io_utils import read_json, sha256_file, sha256_text


@dataclass(frozen=True)
class PairRecord:
    schema: str
    pair_id: str
    semantic_id: str
    batch_name: str
    shard_name: str
    sample_name: str
    subject_id: int | None
    motion_dir: str
    caption_json: str
    action_text: str
    action_joiner: str
    selected_segment_indices: list[int]
    selected_segments: list[dict[str, Any]]
    selection_rule: str
    caption_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _subject_from_sample_name(name: str) -> int | None:
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else None


def _segment_sort_key(item: tuple[int, dict[str, Any]]) -> tuple[float, float, int]:
    index, segment = item
    return (float(segment.get("start_time", 0.0)), float(segment.get("end_time", 0.0)), index)


def build_pair_record(caption_path: Path, motion_dir: Path, batch_name: str, shard_name: str) -> PairRecord:
    raw = read_json(caption_path)
    if not isinstance(raw, list):
        raise ValueError(f"Caption must be a JSON list: {caption_path}")
    subject_id = _subject_from_sample_name(caption_path.stem)
    indexed = [(i, segment) for i, segment in enumerate(raw) if isinstance(segment, dict)]
    subject_segments = [(i, segment) for i, segment in indexed if segment.get("subject_id") == subject_id]
    if subject_id is not None and subject_segments:
        selected = subject_segments
        rule = "filename_subject_id_match"
    else:
        selected = indexed
        rule = "all_segments_fallback"
    selected = sorted(selected, key=_segment_sort_key)
    selected = [(i, segment) for i, segment in selected if str(segment.get("action", "")).strip()]
    if not selected:
        raise ValueError(f"No non-empty action field in {caption_path}")
    actions = [str(segment["action"]).strip() for _, segment in selected]
    action_text = " Then, ".join(actions)
    normalized = " ".join(action_text.lower().split())
    pair_id = f"{batch_name}/{shard_name}/{caption_path.stem}"
    return PairRecord(
        schema="audio_motion_pair.v1",
        pair_id=pair_id,
        semantic_id=sha256_text(normalized),
        batch_name=batch_name,
        shard_name=shard_name,
        sample_name=caption_path.stem,
        subject_id=subject_id,
        motion_dir=str(motion_dir.resolve()),
        caption_json=str(caption_path.resolve()),
        action_text=action_text,
        action_joiner=" Then, ",
        selected_segment_indices=[index for index, _ in selected],
        selected_segments=[segment for _, segment in selected],
        selection_rule=rule,
        caption_sha256=sha256_file(caption_path),
    )


def iter_pairs(batch_root: Path) -> Iterator[PairRecord]:
    caption_root = batch_root / "motion_caption"
    motion_root = batch_root / "converted_motion"
    if not caption_root.is_dir() or not motion_root.is_dir():
        raise FileNotFoundError(f"Expected motion_caption/ and converted_motion/ under {batch_root}")
    batch_name = batch_root.name
    for shard_dir in sorted(path for path in caption_root.iterdir() if path.is_dir()):
        motion_shard = motion_root / shard_dir.name
        if not motion_shard.is_dir():
            raise FileNotFoundError(f"Missing motion shard: {motion_shard}")
        for caption_path in sorted(shard_dir.glob("*.json")):
            motion_dir = motion_shard / caption_path.stem
            if not motion_dir.is_dir():
                raise FileNotFoundError(f"Missing paired motion directory: {motion_dir}")
            yield build_pair_record(caption_path, motion_dir, batch_name, shard_dir.name)
