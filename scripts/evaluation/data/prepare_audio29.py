#!/usr/bin/env python3
"""Pre-warm the automatic rhythm audio29 cache in parallel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

from scripts.evaluation.data.audio import (
    AUDIO_SUFFIXES,
    FEATURE_FPS,
    extract_audio29,
    _needs_refresh,
    _sample_id,
    materialize_audio29_manifest,
)


def _prepare_one(args: tuple[str, str]) -> str:
    audio_name, feature_name = args
    audio_path = Path(audio_name)
    feature_path = Path(feature_name)
    if _needs_refresh(feature_path, audio_path):
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature = extract_audio29(audio_path)
        import numpy as np

        np.save(feature_path, feature, allow_pickle=False)
    return audio_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="rhythm root containing raw/ or audio files")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "raw" if (root / "raw").is_dir() else root
    from scripts.evaluation.data.audio import FEATURE_DIRECTORY

    output_root = root / FEATURE_DIRECTORY
    paths = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no supported audio files under {source_root}")
    ids = [_sample_id(path) for path in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate normalized sample IDs in rhythm audio source")
    jobs = [(str(path), str(output_root / f"{sample_id}.npy")) for path, sample_id in zip(paths, ids, strict=True)]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for _ in tqdm(pool.map(_prepare_one, jobs), total=len(jobs), desc="Preparing audio29", unit="audio"):
            pass
    manifest = materialize_audio29_manifest(root)
    print(f"prepared {len(paths)} audio29 features; manifest={manifest}")


if __name__ == "__main__":
    main()
