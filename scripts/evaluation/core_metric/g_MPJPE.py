"""OMG Video/SONIC global MPJPE for G1 motions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np

from scripts.evaluation.preprocessing.temporal import resample_qpos
from scripts.evaluation.omg_paths import omg_root


OMG_SOURCE_ROOT = omg_root() / "src"
G1_KINEMATICS_PATH = omg_root() / "assets" / "robots" / "g1" / "g1_kinematics.json"

# Official OMG Video/SONIC tracking evaluates this exact 14-link subset.
OMG_SONIC_BODY_INDICES = np.asarray(
    [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29], dtype=np.int64
)


@dataclass(frozen=True)
class GlobalMPJPEScore:
    """One complete prediction--GT motion pair's global MPJPE, in millimetres."""

    value: float
    num_frames: int
    num_joints: int
    duration_seconds: float
    groundtruth_hold_frames: int


class GlobalMPJPEEvaluator:
    """G1 FK adapter for OMG Video/SONIC global MPJPE.

    The protocol compares the official 14-link subset directly in world
    coordinates and reports millimetres. No root translation, root rotation,
    rigid, or Procrustes alignment is applied.
    """

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
        self.body_order = tuple(self.kinematics.body_order)
        self.body_indices = tuple(int(index) for index in OMG_SONIC_BODY_INDICES)

    def score(
        self,
        prediction_qpos_36: np.ndarray,
        groundtruth_qpos_36: np.ndarray,
        *,
        prediction_fps: float,
        groundtruth_fps: float,
        target_fps: float,
        groundtruth_hold_frames: int,
    ) -> GlobalMPJPEScore:
        prediction, groundtruth, duration = strict_omg_sonic_tracking_qpos(
            prediction_qpos_36,
            groundtruth_qpos_36,
            prediction_fps=prediction_fps,
            groundtruth_fps=groundtruth_fps,
            expected_fps=target_fps,
            groundtruth_hold_frames=groundtruth_hold_frames,
        )
        torch = self.torch
        with torch.inference_mode():
            qpos = torch.as_tensor(
                np.stack((prediction, groundtruth)), dtype=torch.float32, device=self.device
            )
            body_pos_w = self.kinematics.forward_kinematics(qpos)["body_pos_w"]
            body_indices = torch.as_tensor(self.body_indices, device=self.device)
            tracked_positions = body_pos_w.index_select(dim=2, index=body_indices)
            errors = torch.linalg.vector_norm(tracked_positions[0] - tracked_positions[1], dim=-1)
        return GlobalMPJPEScore(
            value=float(errors.mean().detach().cpu()) * 1000.0,
            num_frames=int(errors.shape[0]),
            num_joints=int(errors.shape[1]),
            duration_seconds=duration,
            groundtruth_hold_frames=int(groundtruth_hold_frames),
        )


def groundtruth_start_hold_frames(clip: Path) -> int:
    """Read the required Video/SONIC GT initial-hold length from ``info.txt``."""
    info_path = Path(clip) / "info.txt"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"OMG Video/SONIC tracking requires ground-truth info.txt: {info_path}"
        )
    match = re.search(
        r"^start_hold_frames:\s*(\d+)\s*$", info_path.read_text(encoding="utf-8"), re.MULTILINE
    )
    if match is None:
        raise ValueError(f"missing start_hold_frames in {info_path}")
    return int(match.group(1))


def strict_omg_sonic_tracking_qpos(
    prediction_qpos_36: np.ndarray,
    groundtruth_qpos_36: np.ndarray,
    *,
    prediction_fps: float,
    groundtruth_fps: float,
    expected_fps: float = 50.0,
    groundtruth_hold_frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Remove GT hold frames, then require strict native-50-Hz equality.

    The official tracking protocol does not truncate, interpolate, pad, or use
    task duration to repair a mismatched pair.
    """
    prediction = np.asarray(prediction_qpos_36, dtype=np.float32)
    groundtruth = np.asarray(groundtruth_qpos_36, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[1] != 36:
        raise ValueError(f"prediction qpos must have shape (T, 36), got {prediction.shape}")
    if groundtruth.ndim != 2 or groundtruth.shape[1] != 36:
        raise ValueError(f"groundtruth qpos must have shape (T, 36), got {groundtruth.shape}")
    if not np.isclose(float(expected_fps), 50.0):
        raise ValueError("OMG Video/SONIC tracking protocol is fixed at 50 FPS")
    if not np.isclose(float(prediction_fps), float(expected_fps)):
        raise ValueError(f"prediction FPS must be {expected_fps:g} for OMG tracking, got {prediction_fps:g}")
    if not np.isclose(float(groundtruth_fps), float(expected_fps)):
        raise ValueError(f"groundtruth FPS must be {expected_fps:g} for OMG tracking, got {groundtruth_fps:g}")
    hold = int(groundtruth_hold_frames)
    if hold < 0:
        raise ValueError(f"groundtruth_hold_frames must be non-negative, got {hold}")
    if hold >= len(groundtruth):
        raise ValueError(
            f"groundtruth_hold_frames={hold} removes all {len(groundtruth)} ground-truth frames"
        )
    groundtruth = groundtruth[hold:]
    if len(prediction) < 2 or len(groundtruth) < 2:
        raise ValueError("prediction and GT after hold removal each require at least two frames")
    if prediction.shape != groundtruth.shape:
        raise ValueError(
            "Shape mismatch after GT hold removal: "
            f"prediction={prediction.shape}, groundtruth={groundtruth.shape}, hold_frames={hold}"
        )
    return prediction, groundtruth, float((len(prediction) - 1) / expected_fps)


def common_raw_50hz_prefix_qpos(
    prediction_qpos_36: np.ndarray,
    groundtruth_qpos_36: np.ndarray,
    *,
    prediction_fps: float,
    groundtruth_fps: float,
    expected_fps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the shared native-frame prefix required by OMG physical metrics.

    Both inputs must already be exported at the fixed physical evaluation FPS.
    The function intentionally performs no interpolation, resampling, or
    duration-metadata remapping.
    """
    prediction = np.asarray(prediction_qpos_36, dtype=np.float32)
    groundtruth = np.asarray(groundtruth_qpos_36, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[1] != 36:
        raise ValueError(f"prediction qpos must have shape (T, 36), got {prediction.shape}")
    if groundtruth.ndim != 2 or groundtruth.shape[1] != 36:
        raise ValueError(f"groundtruth qpos must have shape (T, 36), got {groundtruth.shape}")
    if len(prediction) < 2 or len(groundtruth) < 2:
        raise ValueError("prediction and groundtruth each require at least two source frames")
    if not np.isclose(float(expected_fps), 50.0):
        raise ValueError("OMG g-MPJPE formal protocol is fixed at 50 FPS")
    if not np.isclose(float(prediction_fps), float(expected_fps)):
        raise ValueError(f"prediction FPS must be {expected_fps:g} for OMG g-MPJPE, got {prediction_fps:g}")
    if not np.isclose(float(groundtruth_fps), float(expected_fps)):
        raise ValueError(f"groundtruth FPS must be {expected_fps:g} for OMG g-MPJPE, got {groundtruth_fps:g}")
    frames = min(len(prediction), len(groundtruth))
    return prediction[:frames], groundtruth[:frames], float((frames - 1) / expected_fps)


def common_time_resample_qpos(
    prediction_qpos_36: np.ndarray,
    groundtruth_qpos_36: np.ndarray,
    *,
    prediction_fps: float,
    groundtruth_fps: float,
    target_fps: float,
    prediction_duration_seconds: float | None = None,
    groundtruth_duration_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample both sequences to identical timestamps on their real-time prefix.

    Prediction time defaults to its observed CSV timeline.  Ground truth may
    instead receive the authoritative OMG task ``metadata.duration``.
    """
    prediction = np.asarray(prediction_qpos_36, dtype=np.float32)
    groundtruth = np.asarray(groundtruth_qpos_36, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[1] != 36:
        raise ValueError(f"prediction qpos must have shape (T, 36), got {prediction.shape}")
    if groundtruth.ndim != 2 or groundtruth.shape[1] != 36:
        raise ValueError(f"groundtruth qpos must have shape (T, 36), got {groundtruth.shape}")
    if len(prediction) < 2 or len(groundtruth) < 2:
        raise ValueError("prediction and groundtruth each require at least two source frames")
    if prediction_fps <= 0 or groundtruth_fps <= 0 or target_fps <= 0:
        raise ValueError("prediction_fps, groundtruth_fps, and target_fps must be positive")

    prediction_duration = (
        (len(prediction) - 1) / float(prediction_fps)
        if prediction_duration_seconds is None else float(prediction_duration_seconds)
    )
    groundtruth_duration = (
        (len(groundtruth) - 1) / float(groundtruth_fps)
        if groundtruth_duration_seconds is None else float(groundtruth_duration_seconds)
    )
    if not np.isfinite(prediction_duration) or prediction_duration <= 0.0:
        raise ValueError("prediction_duration_seconds must be a positive finite number")
    if not np.isfinite(groundtruth_duration) or groundtruth_duration <= 0.0:
        raise ValueError("groundtruth_duration_seconds must be a positive finite number")
    duration = min(prediction_duration, groundtruth_duration)
    frames = int(np.floor(duration * float(target_fps) + 1e-6)) + 1
    if frames < 2:
        raise ValueError("prediction and groundtruth share fewer than two frames on the target timeline")
    prediction_resampled = resample_qpos(
        prediction, prediction_fps, target_fps, duration_seconds=prediction_duration
    )[:frames]
    groundtruth_resampled = resample_qpos(
        groundtruth, groundtruth_fps, target_fps, duration_seconds=groundtruth_duration
    )[:frames]
    if len(prediction_resampled) != frames or len(groundtruth_resampled) != frames:
        raise RuntimeError("common-time resampling produced an incomplete target timeline")
    return prediction_resampled, groundtruth_resampled, float((frames - 1) / target_fps)


def _resolve_device(requested: str, torch_module: Any):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)
