from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ProjectConfig
from .encoders import FrozenOMGMotionEncoder, FrozenWhisperSpeechEncoder
from .io_utils import sha256_file
from .motion import make_windows, resample_qpos
from .objective import AudioMotionAlignment


class AudioMotionEvaluator:
    """Inference-only Audio-Motion evaluator using the trained V1 alignment."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        omg_root: Path,
        omg_checkpoint: Path,
        device: torch.device,
        whisper_model_path: Path | None = None,
    ) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("schema") not in {
            "audio_motion_evaluator_checkpoint.v1",
            "audio_motion_evaluator_inference.v1",
        }:
            raise ValueError(f"Unsupported evaluator checkpoint schema in {checkpoint_path}")
        architecture = state["architecture"]
        expected_omg_hash = str(architecture["omg_checkpoint_sha256"])
        actual_omg_hash = sha256_file(omg_checkpoint)
        if actual_omg_hash != expected_omg_hash:
            raise ValueError(
                "OMG motion checkpoint does not match the checkpoint used for Audio-Motion training: "
                f"expected {expected_omg_hash}, got {actual_omg_hash}"
            )

        config = replace(
            ProjectConfig(),
            omg_root=omg_root.expanduser().resolve(),
            omg_checkpoint=omg_checkpoint.expanduser().resolve(),
            whisper_model=str(whisper_model_path or architecture["speech_backbone"]),
            whisper_revision=str(architecture["speech_backbone_revision"]),
            embedding_dim=int(architecture["embedding_dim"]),
        )
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.motion_encoder = FrozenOMGMotionEncoder(config, device)
        self.speech_encoder = FrozenWhisperSpeechEncoder(config, device)
        self.alignment = AudioMotionAlignment(
            int(architecture["audio_feature_dim"]), int(architecture["embedding_dim"])
        ).to(device)
        self.alignment.load_state_dict(state["audio_motion_alignment"], strict=True)
        self.alignment.requires_grad_(False).eval()
        self.architecture: dict[str, Any] = dict(architecture)
        self.validation = dict(state.get("validation") or {})

    @torch.inference_mode()
    def encode_audio_batch(
        self, audio_paths: list[Path], *, chunk_batch_size: int = 8
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        raw = self.speech_encoder.encode_files(audio_paths, chunk_batch_size=chunk_batch_size)
        features = torch.from_numpy(np.stack([value for value, _ in raw])).to(self.device)
        embeddings = self.alignment(features).cpu().numpy().astype(np.float32)
        return [(embeddings[index], metadata) for index, (_, metadata) in enumerate(raw)]

    @torch.inference_mode()
    def encode_audio(self, audio_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        return self.encode_audio_batch([audio_path], chunk_batch_size=1)[0]

    @torch.inference_mode()
    def encode_motion_batch(
        self, qpos_batch: list[np.ndarray], source_fps: float, *, window_batch_size: int = 128
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        prepared: list[tuple[np.ndarray, np.ndarray, list[Any], np.ndarray, np.ndarray]] = []
        all_windows: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        owners: list[int] = []
        for owner, source in enumerate(qpos_batch):
            qpos = np.asarray(source, dtype=np.float32)
            if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
                raise ValueError(f"qpos_36 must have shape (T,36), T >= 2; got {qpos.shape}")
            if not np.isfinite(qpos).all():
                raise ValueError("qpos_36 contains non-finite values")
            qpos30, target_times = resample_qpos(qpos, float(source_fps), self.config.derived_fps)
            windows, masks, records = make_windows(qpos30, self.config.window_frames, self.config.window_stride)
            prepared.append((qpos, qpos30, records, target_times, masks))
            all_windows.extend(windows)
            all_masks.extend(masks)
            owners.extend([owner] * len(windows))
        sums = np.zeros((len(prepared), self.config.embedding_dim), dtype=np.float32)
        counts = np.zeros(len(prepared), dtype=np.int64)
        for start in range(0, len(all_windows), window_batch_size):
            stop = min(start + window_batch_size, len(all_windows))
            embeddings = self.motion_encoder.encode_window_embeddings(
                np.stack(all_windows[start:stop]), np.stack(all_masks[start:stop])
            ).detach().cpu().numpy().astype(np.float32)
            for row, owner in enumerate(owners[start:stop]):
                sums[owner] += embeddings[row]
                counts[owner] += 1
        if np.any(counts == 0):
            raise RuntimeError("motion batching produced an input without windows")
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise RuntimeError("motion batching produced a zero-norm embedding")
        embeddings = sums / norms
        output: list[tuple[np.ndarray, dict[str, Any]]] = []
        for index, (source, qpos30, records, target_times, _) in enumerate(prepared):
            output.append((embeddings[index].astype(np.float32), {
                "source_fps": float(source_fps),
                "source_frames": int(len(source)),
                "derived_fps": self.config.derived_fps,
                "derived_frames": int(len(qpos30)),
                "derived_duration_seconds": float(target_times[-1]) if len(target_times) else 0.0,
                "resampling": {
                    "translation_and_joints": "linear_interpolation",
                    "root_quaternion": "shortest_path_unit_quaternion_slerp_wxyz_numpy",
                },
                "window_frames": self.config.window_frames,
                "window_stride": self.config.window_stride,
                "window_count": len(records),
                "aggregation": "mean_of_window_embeddings_then_l2_normalize",
            }))
        return output

    @torch.inference_mode()
    def encode_motion(self, qpos_36: np.ndarray, source_fps: float) -> tuple[np.ndarray, dict[str, Any]]:
        qpos = np.asarray(qpos_36, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"qpos_36 must have shape (T,36), got {qpos.shape}")
        qpos30, target_times = resample_qpos(qpos, float(source_fps), self.config.derived_fps)
        windows, masks, records = make_windows(
            qpos30, self.config.window_frames, self.config.window_stride
        )
        embedding = self.motion_encoder.encode_windows(windows, masks)
        metadata = {
            "source_fps": float(source_fps),
            "source_frames": int(len(qpos)),
            "derived_fps": self.config.derived_fps,
            "derived_frames": int(len(qpos30)),
            "derived_duration_seconds": float(target_times[-1]) if len(target_times) else 0.0,
            "resampling": {
                "translation_and_joints": "linear_interpolation",
                "root_quaternion": "shortest_path_unit_quaternion_slerp_wxyz_numpy",
            },
            "window_frames": self.config.window_frames,
            "window_stride": self.config.window_stride,
            "window_count": len(records),
            "aggregation": "mean_of_window_embeddings_then_l2_normalize",
        }
        return embedding, metadata
