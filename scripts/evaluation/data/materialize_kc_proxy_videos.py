"""Materialize sparse KC keyframe JPGs as reproducible 30 FPS proxy MP4 files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_KEYFRAME = re.compile(r"^(?P<sample_id>.+)_joint(?P<index>\d+)(?:_skel)?\.jpg$")


def _metadata_durations(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            sample_id = str(row.get("name", ""))
            duration = float(row.get("duration", 0.0))
            if sample_id and duration > 0.0:
                durations[sample_id] = duration
            elif sample_id:
                raise ValueError(f"metadata line {line_number}: invalid duration for {sample_id!r}")
    return durations


def _keyframe_groups(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in root.glob("*.jpg"):
        match = _KEYFRAME.match(path.name)
        if match is not None:
            groups[match.group("sample_id")].append((int(match.group("index")), path))
    return {
        sample_id: [path for _, path in sorted(items)]
        for sample_id, items in groups.items()
    }


def _load_rgb(path: Path, size: tuple[int, int] | None) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if size is not None and rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        frame = np.asarray(rgb, dtype=np.uint8)
    height, width = frame.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height < 2 or even_width < 2:
        raise ValueError(f"keyframe {path} is too small for yuv420p: {(width, height)}")
    return frame[:even_height, :even_width]


def _write_proxy(
    output_path: Path, keyframes: list[Path], duration: float, fps: int, max_side: int
) -> int:
    total_frames = max(1, int(round(duration * fps)))
    with Image.open(keyframes[0]) as image:
        source_width, source_height = image.size
    scale = min(1.0, max_side / max(source_width, source_height))
    expected_size = (
        max(2, int(round(source_width * scale)) // 2 * 2),
        max(2, int(round(source_height * scale)) // 2 * 2),
    )
    first = _load_rgb(keyframes[0], expected_size)
    frames = [first]
    for path in keyframes[1:]:
        frames.append(_load_rgb(path, expected_size))
    temporary = output_path.with_name(f"{output_path.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = imageio.get_writer(
        temporary,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        for frame_index in range(total_frames):
            source_index = min(len(frames) - 1, frame_index * len(frames) // total_frames)
            writer.append_data(frames[source_index])
    finally:
        writer.close()
    temporary.replace(output_path)
    return total_frames


def _manifest(
    *, task: str, encoder: str, source_root: Path, max_side: int, entries: Iterator[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema": "liujiahui.instruction_manifest.v1",
        "kind": "video",
        "encoder": encoder,
        "format": "proxy_keyframe_video_v1",
        "task": task,
        "source": str(source_root),
        "proxy_policy": {
            "fps": 30,
            "duration_source": "data/metadata.jsonl:name->duration",
            "ordering": "ascending_joint_index",
            "timing": "keyframes_are_held_for_equal_time_intervals",
            "codec": "libx264/yuv420p",
            "max_side_pixels": max_side,
            "raw_keyframes_preserved": True,
        },
        "samples": list(entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.jsonl"))
    parser.add_argument("--task", default="kc")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.fps <= 0 or args.max_side < 2:
        parser.error("--fps must be positive and --max-side must be at least 2")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    proxy_root = output_root / "preprocessed"
    proxy_root.mkdir(parents=True, exist_ok=True)
    groups = _keyframe_groups(input_root)
    durations = _metadata_durations(args.metadata.resolve())
    sample_ids = sorted(groups)
    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]

    completed: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for sample_id in tqdm(sample_ids, desc=f"Proxy MP4 {args.encoder}", unit="sample"):
        try:
            duration = durations[sample_id]
            output_path = proxy_root / f"{sample_id}.mp4"
            keyframes = groups[sample_id]
            if args.overwrite or not output_path.is_file():
                frame_count = _write_proxy(output_path, keyframes, duration, args.fps, args.max_side)
            else:
                frame_count = max(1, int(round(duration * args.fps)))
            completed.append({
                "sample_id": sample_id,
                "path": str(output_path.relative_to(output_root)),
                "duration_seconds": duration,
                "proxy_frames": frame_count,
                "keyframe_count": len(keyframes),
            })
        except Exception as exc:
            failures[sample_id] = str(exc)

    if failures:
        raise RuntimeError(f"failed {len(failures)} proxy videos; examples: {list(failures.items())[:3]}")
    manifest = _manifest(
        task=args.task,
        encoder=args.encoder,
        source_root=input_root,
        max_side=args.max_side,
        entries=iter(completed),
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "completed": len(completed),
        "proxy_root": str(proxy_root),
        "manifest": str(output_root / "manifest.json"),
    }))


if __name__ == "__main__":
    main()
