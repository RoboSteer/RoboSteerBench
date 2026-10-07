"""OMG-formal 29-D audio features used by the MUL_RHY BAS protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


AUDIO_SUFFIXES = frozenset((".mp3", ".wav"))
FEATURE_FPS = 50
FEATURE_N_FFT = 1024
FEATURE_PROTOCOL = "OMG.prepare_mul_rhy_omg_artifacts.audio29"
FEATURE_DIRECTORY = f"audio29_omg_{FEATURE_FPS}fps"


def materialize_audio29_manifest(root: Path) -> Path:
    """Create/reuse the exact OMG 50 Hz audio29 cache and its manifest.

    The final feature channel is the binary music-beat signal. A separate
    cache directory prevents a legacy LDA ``n_fft=2048`` cache from ever being
    mistaken for the OMG formal protocol.
    """
    root = root.resolve()
    source_root = root / "raw" if (root / "raw").is_dir() else root
    audio_paths = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not audio_paths:
        raise FileNotFoundError(
            f"no .mp3/.wav files under {source_root}; provide a rhythm directory with raw/"
        )
    output_root = root / FEATURE_DIRECTORY
    output_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, str]] = []
    seen: set[str] = set()
    for audio_path in tqdm(audio_paths, desc="Preparing OMG audio29", unit="audio", dynamic_ncols=True):
        sample_id = _sample_id(audio_path)
        if sample_id in seen:
            raise ValueError(f"duplicate rhythm sample id {sample_id!r} under {source_root}")
        seen.add(sample_id)
        feature_path = output_root / f"{sample_id}.npy"
        if _needs_refresh(feature_path, audio_path):
            np.save(feature_path, extract_audio29(audio_path), allow_pickle=False)
        samples.append({"sample_id": sample_id, "path": str(feature_path.relative_to(root))})

    manifest_path = root / "manifest.json"
    payload = {
        "encoder": "audio_features",
        "kind": "audio29",
        "feature_fps": FEATURE_FPS,
        "feature_protocol": FEATURE_PROTOCOL,
        "feature_n_fft": FEATURE_N_FFT,
        "raw_root": str(source_root),
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def extract_audio29(path: Path, fps: float = FEATURE_FPS) -> np.ndarray:
    """Copy the official OMG ``audio29`` implementation without alteration."""
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("librosa is required for OMG MUL_RHY BAS extraction") from exc
    waveform, sample_rate = librosa.load(path, sr=16000, mono=True)
    hop = int(round(sample_rate / float(fps)))
    if hop <= 0:
        raise ValueError("audio FPS must be positive")
    magnitude = np.abs(librosa.stft(
        waveform, n_fft=1024, hop_length=hop, win_length=1024, center=False
    )).astype(np.float32)
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=20, n_fft=1024,
        hop_length=hop, win_length=1024, center=False,
    ).astype(np.float32)
    chroma = librosa.feature.chroma_stft(S=magnitude, sr=sample_rate, norm=None).astype(np.float32)
    flux = np.zeros(magnitude.shape[1], dtype=np.float32)
    if len(flux) > 1:
        flux[1:] = np.maximum(0.0, magnitude[:, 1:] - magnitude[:, :-1]).sum(axis=0)
    onset = librosa.onset.onset_strength(
        y=waveform, sr=sample_rate, hop_length=hop, center=False
    ).astype(np.float32)
    beat = np.zeros_like(onset)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sample_rate, hop_length=hop, units="frames"
    )
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    beat[beat_frames[(beat_frames >= 0) & (beat_frames < len(beat))]] = 1.0
    frames = min(mfcc.shape[1], chroma.shape[1], len(flux), len(onset), len(beat))
    if frames < 2:
        raise ValueError(f"audio is too short for BAS: {path}")
    value = np.concatenate((
        mfcc[:, :frames].T,
        (chroma[:6, :frames] + chroma[6:, :frames]).T,
        flux[:frames, None], onset[:frames, None], beat[:frames, None],
    ), axis=1).astype(np.float32)
    if value.shape[1] != 29 or not np.isfinite(value).all():
        raise ValueError(f"invalid OMG audio29 features extracted from {path}: {value.shape}")
    return value


def _sample_id(audio_path: Path) -> str:
    return audio_path.stem.removesuffix("_music").removesuffix("_audio")


def _needs_refresh(feature_path: Path, audio_path: Path) -> bool:
    if not feature_path.is_file():
        return True
    try:
        return feature_path.stat().st_mtime_ns < audio_path.stat().st_mtime_ns
    except OSError:
        return True
