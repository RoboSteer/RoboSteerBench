"""Manifest-driven instruction input loading with automatic rhythm-audio support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InstructionSample:
    sample_id: str
    path: Path
    key: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstructionIndex:
    root: Path
    encoder: str
    samples: dict[str, InstructionSample]
    metadata: dict[str, Any]


def load_instruction_index(path: Path) -> InstructionIndex:
    """Read a manifest or materialize a supported raw instruction directory.

    Known video-task layouts and rhythm-audio roots can be materialized
    automatically. Other directories fail with a modality-specific diagnostic
    instead of silently treating videos as audio.
    """
    path = path.resolve()
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        if not path.is_dir():
            raise FileNotFoundError(
                f"instruction manifest not found: {manifest_path}. "
                "Provide a manifest JSON or a directory containing MP3/WAV files."
            )
        video_suffixes = frozenset((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        audio_suffixes = frozenset((".mp3", ".wav"))
        has_video = any(
            item.is_file() and item.suffix.lower() in video_suffixes
            for item in path.rglob("*")
        )
        has_audio = any(
            item.is_file() and item.suffix.lower() in audio_suffixes
            for item in path.rglob("*")
        )
        if has_video and has_audio:
            raise ValueError(
                f"instruction directory mixes video and audio without a manifest: {path}"
            )
        if has_video:
            from scripts.evaluation.data.materialize_video_manifests import materialize_video_manifest

            manifest_path = materialize_video_manifest(path)
        elif has_audio:
            from scripts.evaluation.data.audio import materialize_audio29_manifest

            manifest_path = materialize_audio29_manifest(path)
        else:
            raise FileNotFoundError(
                f"instruction directory has no manifest and no supported video/audio files: {path}"
            )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A rhythm directory may contain a legacy LDA cache.  If raw audio is
    # available, promote it automatically to the formal OMG feature protocol.
    if path.is_dir() and (path / "raw").is_dir():
        from scripts.evaluation.data.audio import FEATURE_PROTOCOL, materialize_audio29_manifest

        if payload.get("kind") == "audio29" and payload.get("feature_protocol") != FEATURE_PROTOCOL:
            manifest_path = materialize_audio29_manifest(path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder = payload.get("encoder")
    if not isinstance(encoder, str) or not encoder:
        raise ValueError(f"{manifest_path} must contain a non-empty 'encoder' field")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{manifest_path} must contain a 'samples' list")
    root = manifest_path.parent
    samples: dict[str, InstructionSample] = {}
    for row in raw_samples:
        sample_id = str(row["sample_id"])
        # Text manifests use an inline ``text`` field instead of an external
        # tensor/audio file.  Keep a stable, existing manifest path in the
        # generic dataclass; text adapters read ``attributes["text"]``.
        if "path" in row:
            sample_path = Path(row["path"])
            if not sample_path.is_absolute():
                sample_path = root / sample_path
        elif payload.get("format") == "inline_text_v1" and isinstance(row.get("text"), str):
            sample_path = manifest_path
        else:
            raise ValueError(
                f"instruction sample {sample_id!r} in {manifest_path} needs 'path', "
                "or an inline_text_v1 non-empty 'text' field"
            )
        if sample_id in samples:
            raise ValueError(f"duplicate instruction sample_id {sample_id!r} in {manifest_path}")
        attributes = {
            str(key): value for key, value in row.items()
            if key not in {"sample_id", "path", "key", "assets"}
        }
        if isinstance(row.get("assets"), dict):
            attributes["assets"] = dict(row["assets"])
        samples[sample_id] = InstructionSample(
            sample_id, sample_path.resolve(), row.get("key"), attributes
        )
    return InstructionIndex(root=root, encoder=encoder, samples=samples, metadata=payload)


def load_instruction_array(sample: InstructionSample) -> np.ndarray:
    """Load a numeric instruction tensor from NPY, NPZ, or CSV."""
    if not sample.path.is_file():
        raise FileNotFoundError(f"instruction input does not exist: {sample.path}")
    suffix = sample.path.suffix.lower()
    if suffix == ".npy":
        value = np.load(sample.path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(sample.path, allow_pickle=False) as payload:
            key = sample.key
            if key is None:
                keys = list(payload.files)
                if len(keys) != 1:
                    raise ValueError(f"{sample.path} contains {keys}; specify 'key' in the manifest")
                key = keys[0]
            value = payload[key]
    elif suffix == ".csv":
        value = np.loadtxt(sample.path, delimiter=",", dtype=np.float32, ndmin=2)
    else:
        raise ValueError(f"unsupported instruction input {sample.path}; expected NPY, NPZ, or CSV")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite instruction values in {sample.path}")
    return value
