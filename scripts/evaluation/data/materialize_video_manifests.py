"""Build deterministic instruction manifests for the repository's video tasks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "liujiahui.instruction_manifest.v1"
VIDEO_SUFFIXES = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm"))
SEGMENT_AGGREGATION = "ordered_segment_embeddings_l2_normalized_mean_v1"


@dataclass(frozen=True)
class Layout:
    pattern: re.Pattern[str]
    segment_order: tuple[str, ...] = ()


_LAYOUTS = {
    ("fore", "human"): Layout(re.compile(r"(?P<sample_id>.+)_fore_input\.mp4")),
    ("fore", "skel"): Layout(re.compile(r"(?P<sample_id>.+)_fore_skel_input\.mp4")),
    ("retro", "human"): Layout(re.compile(r"(?P<sample_id>.+)_retro_input\.mp4")),
    ("retro", "skel"): Layout(re.compile(r"(?P<sample_id>.+)_retro_skel_input\.mp4")),
    ("upper-full", "human"): Layout(re.compile(r"(?P<sample_id>.+)_no_upper\.mp4")),
    ("upper-full", "skel"): Layout(re.compile(r"(?P<sample_id>.+)_no_upper_skel\.mp4")),
    ("lower-full", "human"): Layout(re.compile(r"(?P<sample_id>.+)_no_lower\.mp4")),
    ("lower-full", "skel"): Layout(re.compile(r"(?P<sample_id>.+)_no_lower_skel\.mp4")),
    ("inter", "human"): Layout(
        re.compile(r"(?P<sample_id>.+)_(?P<segment>start|end)\.mp4"),
        ("start", "end"),
    ),
    ("inter", "skel"): Layout(
        re.compile(r"(?P<sample_id>.+)_(?P<segment>start|end)_skel\.mp4"),
        ("start", "end"),
    ),
}


def materialize_video_manifest(root: Path, *, write: bool = True) -> Path:
    """Validate a known video-task directory and create its manifest atomically."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"video instruction directory does not exist: {root}")
    task = root.parent.name
    video_kind = root.name
    layout = _LAYOUTS.get((task, video_kind))
    if layout is None:
        supported = ", ".join(f"{task}/{kind}" for task, kind in sorted(_LAYOUTS))
        raise ValueError(
            f"cannot infer a video manifest layout for {root}; supported layouts: {supported}"
        )

    videos = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"no supported video files under {root}")

    grouped: dict[str, dict[str, Path]] = {}
    unexpected: list[str] = []
    for path in videos:
        match = layout.pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        sample_id = match.group("sample_id")
        segment = match.groupdict().get("segment") or "video"
        existing = grouped.setdefault(sample_id, {}).get(segment)
        if existing is not None:
            raise ValueError(
                f"duplicate {segment!r} video for sample {sample_id!r}: {existing}, {path}"
            )
        grouped[sample_id][segment] = path
    if unexpected:
        raise ValueError(
            f"{root} contains {len(unexpected)} videos that do not match the {task}/{video_kind} "
            f"layout; examples: {unexpected[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        segments = grouped[sample_id]
        if layout.segment_order:
            missing = [name for name in layout.segment_order if name not in segments]
            extra = sorted(set(segments) - set(layout.segment_order))
            if missing or extra:
                raise ValueError(
                    f"sample {sample_id!r} has invalid video segments; "
                    f"missing={missing}, extra={extra}"
                )
            paths = [segments[name].name for name in layout.segment_order]
            rows.append({
                "sample_id": sample_id,
                "path": paths[0],
                "assets": {"video": paths},
                "segment_order": list(layout.segment_order),
            })
        else:
            rows.append({"sample_id": sample_id, "path": segments["video"].name})

    human = video_kind == "human"
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "video",
        "encoder": "video_motion_human" if human else "video_motion_skel",
        "format": (
            "ordered_relative_video_segments_v1"
            if layout.segment_order else "relative_video_path_v1"
        ),
        "task": task,
        "video_kind": "human" if human else "skeleton",
        "condition_modalities": ["human_video" if human else "skeleton_video"],
        "samples": rows,
    }
    if layout.segment_order:
        payload["condition_aggregation"] = SEGMENT_AGGREGATION
        payload["segment_order"] = list(layout.segment_order)

    manifest_path = root / "manifest.json"
    if write:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    return manifest_path


def known_video_roots(data_root: Path) -> list[Path]:
    return [
        data_root / task / video_kind
        for task, video_kind in sorted(_LAYOUTS)
        if (data_root / task / video_kind).is_dir()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "video",
    )
    parser.add_argument("--check", action="store_true", help="validate without writing manifests")
    args = parser.parse_args()
    roots = known_video_roots(args.data_root.resolve())
    if not roots:
        raise RuntimeError(f"no known video-task directories under {args.data_root}")
    for root in roots:
        manifest = materialize_video_manifest(root, write=not args.check)
        action = "validated" if args.check else "wrote"
        print(f"{action} {manifest}")


if __name__ == "__main__":
    main()
