"""Beat-alignment score between an audio condition and generated G1 motion.

The implementation follows the AIST++/Bailando/EDGE convention used by OMG:
music beats are matched to the closest motion beat, where a motion beat is a
local minimum of the whole-body kinetic displacement signal.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from scripts.evaluation.preprocessing.temporal import resample_qpos
from scripts.evaluation.omg_paths import omg_root


OMG_SOURCE_ROOT = omg_root() / "src"
G1_KINEMATICS_PATH = omg_root() / "assets" / "robots" / "g1" / "g1_kinematics.json"


@dataclass(frozen=True)
class BASGenScore:
    """Beat-alignment result for one prediction/audio pair."""

    value: float
    num_audio_beats: int
    num_motion_beats: int
    motion_frames: int
    audio_frames: int


def audio_beats_from_features(audio_features: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Return beat-frame indices from the final audio-feature channel."""
    features = np.asarray(audio_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(f"audio_features must have shape (T, D), T >= 2, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("audio_features must be finite")
    return np.flatnonzero(features[:, -1] > float(threshold)).astype(np.float64)


def motion_beats_from_positions(
    motion_positions: np.ndarray,
    *,
    fps: float,
    min_distance_seconds: float = 0.25,
) -> np.ndarray:
    """Detect motion beats as local minima of mean body displacement."""
    positions = np.asarray(motion_positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[0] < 2 or positions.shape[1] < 2 or positions.shape[2] != 3:
        raise ValueError(f"motion_positions must have shape (T, J, 3), got {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("motion_positions must be finite")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean(axis=-1)
    distance = max(int(round(float(min_distance_seconds) * float(fps))), 1)
    return find_peaks(-displacement, distance=distance)[0].astype(np.float64)


def beat_align_from_beats(
    *,
    audio_beats: np.ndarray,
    motion_beats: np.ndarray,
    motion_fps: float,
    audio_fps: float,
    sigma_frames: float = 3.0,
) -> float:
    """Match every music beat to its nearest motion beat with a Gaussian kernel."""
    if motion_fps <= 0.0 or audio_fps <= 0.0 or sigma_frames <= 0.0:
        raise ValueError("motion_fps, audio_fps, and sigma_frames must be positive")
    audio = np.asarray(audio_beats, dtype=np.float64).reshape(-1)
    motion = np.asarray(motion_beats, dtype=np.float64).reshape(-1)
    if audio.size == 0 or motion.size == 0:
        return 0.0
    audio_times = audio / float(audio_fps)
    motion_times = motion / float(motion_fps)
    nearest_seconds = np.abs(audio_times[:, None] - motion_times[None, :]).min(axis=1)
    sigma_seconds = float(sigma_frames) / float(motion_fps)
    return float(np.exp(-(nearest_seconds**2) / (2.0 * sigma_seconds**2)).mean())


def bas_gen_from_positions(
    *,
    audio_features: np.ndarray,
    motion_positions: np.ndarray,
    motion_fps: float,
    audio_fps: float | None = None,
    beat_threshold: float = 0.5,
    sigma_frames: float = 3.0,
    min_motion_beat_distance_seconds: float = 0.25,
) -> BASGenScore:
    """Compute BAS-Gen from already-FKed body positions and audio features."""
    audio_fps = float(motion_fps if audio_fps is None else audio_fps)
    positions, features = _common_time_prefix(
        np.asarray(motion_positions),
        np.asarray(audio_features),
        motion_fps=float(motion_fps),
        audio_fps=audio_fps,
    )
    audio_beats = audio_beats_from_features(features, threshold=beat_threshold)
    motion_beats = motion_beats_from_positions(
        positions,
        fps=float(motion_fps),
        min_distance_seconds=min_motion_beat_distance_seconds,
    )
    return BASGenScore(
        value=beat_align_from_beats(
            audio_beats=audio_beats,
            motion_beats=motion_beats,
            motion_fps=float(motion_fps),
            audio_fps=audio_fps,
            sigma_frames=sigma_frames,
        ),
        num_audio_beats=int(audio_beats.size),
        num_motion_beats=int(motion_beats.size),
        motion_frames=int(positions.shape[0]),
        audio_frames=int(features.shape[0]),
    )


class BASGenEvaluator:
    """G1 FK adapter around :func:`bas_gen_from_positions`."""

    def __init__(self, *, device: str = "auto") -> None:
        import torch

        if not G1_KINEMATICS_PATH.is_file():
            raise FileNotFoundError(f"G1 kinematics asset is missing: {G1_KINEMATICS_PATH}")
        if str(OMG_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(OMG_SOURCE_ROOT))
        from omg.robots.g1.kinematics import G1Kinematics

        self.torch = torch
        self.device = _resolve_device(device, torch)
        self.kinematics = G1Kinematics(kinematics_path=str(G1_KINEMATICS_PATH)).to(self.device).eval()

    def score(
        self,
        qpos_36: np.ndarray,
        audio_features: np.ndarray,
        *,
        motion_fps: float,
        audio_fps: float,
        beat_threshold: float = 0.5,
        sigma_frames: float = 3.0,
        min_motion_beat_distance_seconds: float = 0.25,
        motion_duration_seconds: float | None = None,
    ) -> BASGenScore:
        qpos = np.asarray(qpos_36, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != 36:
            raise ValueError(f"qpos_36 must have shape (T, 36), T >= 2, got {qpos.shape}")
        if motion_duration_seconds is not None:
            # CSV rows have no timestamps.  For GT, put rows on the
            # task-defined timeline before beat extraction so motion_fps has a
            # physical interpretation rather than merely reflecting row count.
            qpos = resample_qpos(
                qpos, motion_fps, motion_fps, duration_seconds=motion_duration_seconds
            )
        with self.torch.inference_mode():
            qpos_tensor = self.torch.as_tensor(qpos, device=self.device).unsqueeze(0)
            body_positions = self.kinematics.forward_kinematics(qpos_tensor)["body_pos_w"][0].detach().cpu().numpy()
        return bas_gen_from_positions(
            audio_features=audio_features,
            motion_positions=body_positions,
            motion_fps=motion_fps,
            audio_fps=audio_fps,
            beat_threshold=beat_threshold,
            sigma_frames=sigma_frames,
            min_motion_beat_distance_seconds=min_motion_beat_distance_seconds,
        )


def _common_time_prefix(
    motion_positions: np.ndarray,
    audio_features: np.ndarray,
    *,
    motion_fps: float,
    audio_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if motion_fps <= 0.0 or audio_fps <= 0.0:
        raise ValueError("motion_fps and audio_fps must be positive")
    if motion_positions.ndim != 3 or audio_features.ndim != 2:
        raise ValueError("motion_positions and audio_features must be (T, J, 3) and (T, D)")
    duration = min((len(motion_positions) - 1) / motion_fps, (len(audio_features) - 1) / audio_fps)
    motion_frames = int(np.floor(duration * motion_fps + 1e-6)) + 1
    audio_frames = int(np.floor(duration * audio_fps + 1e-6)) + 1
    if motion_frames < 2 or audio_frames < 2:
        raise ValueError("motion and audio must share at least two frames in time")
    return motion_positions[:motion_frames], audio_features[:audio_frames]


def _resolve_device(requested: str, torch_module: Any):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)
