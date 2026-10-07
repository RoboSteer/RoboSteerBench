from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from .model import RotationMotionAlignment
from .omg_encoder import FrozenOMGMotionEncoder
from .preprocessing import load_rotation_condition, load_rotation_target, qpos_to_rotation_features
from .runtime import make_windows, resample_qpos, sha256_file


class RotationMotionEvaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        omg_root: Path,
        omg_checkpoint: Path,
        device: torch.device | str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.omg_root = Path(omg_root).resolve()
        self.omg_checkpoint = Path(omg_checkpoint).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        omg_source = str(self.omg_root / "src")
        if omg_source not in sys.path:
            sys.path.insert(0, omg_source)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "rotation_motion_alignment" not in state:
            raise ValueError(f"Invalid Rotation-Motion checkpoint: {self.checkpoint_path}")
        architecture = state.get("architecture") or {}
        expected_omg_hash = architecture.get("omg_checkpoint_sha256")
        actual_omg_hash = sha256_file(self.omg_checkpoint)
        if expected_omg_hash and actual_omg_hash != expected_omg_hash:
            raise ValueError("OMG evaluator checkpoint SHA256 does not match the training checkpoint")
        self.derived_fps = 30
        self.window_frames = 60
        self.window_stride = 30
        self.model = RotationMotionAlignment(
            int(architecture.get("rotation_feature_dim", 38)),
            int(architecture.get("rotation_temporal_embedding_dim", 256)),
            int(architecture.get("embedding_dim", 512)),
        ).to(self.device)
        self.model.load_state_dict(state["rotation_motion_alignment"], strict=True)
        self.model.requires_grad_(False).eval()
        self.motion_encoder = FrozenOMGMotionEncoder(
            self.omg_root,
            self.omg_checkpoint,
            self.device,
            embedding_dim=int(architecture.get("embedding_dim", 512)),
        )
        self.checkpoint_metadata = {
            "architecture": architecture,
            "validation": state.get("validation"),
            "training_summary": state.get("training_summary") or state.get("training"),
        }

    @torch.inference_mode()
    def encode_qpos(self, qpos_wxyz: np.ndarray, source_fps: float) -> tuple[np.ndarray, dict[str, Any]]:
        qpos = np.asarray(qpos_wxyz, dtype=np.float32)
        qpos_30fps, frame_times = resample_qpos(qpos, source_fps, self.derived_fps)
        rotation_features = qpos_to_rotation_features(qpos_30fps)
        qpos_windows, valid_masks, window_records = make_windows(
            qpos_30fps, self.window_frames, self.window_stride
        )
        rotation_windows, _, _ = make_windows(
            rotation_features, self.window_frames, self.window_stride
        )
        omg_embedding = self.motion_encoder.encode_windows(qpos_windows, valid_masks)
        embedding = self.model.encode(
            torch.from_numpy(omg_embedding).to(self.device)[None],
            torch.from_numpy(rotation_windows).to(self.device)[None],
            torch.from_numpy(valid_masks).to(self.device)[None],
            torch.ones((1, len(rotation_windows)), dtype=torch.bool, device=self.device),
        )[0].cpu().numpy().astype(np.float32)
        metadata = {
            "source_fps": float(source_fps),
            "source_frames": int(len(qpos)),
            "derived_fps": self.derived_fps,
            "derived_frames": int(len(qpos_30fps)),
            "derived_frame_times": frame_times.tolist(),
            "window_frames": self.window_frames,
            "window_stride": self.window_stride,
            "windows": [asdict(item) for item in window_records],
            "qpos_layout": "root_xyz + root_quaternion_wxyz + 29_joint_angles",
        }
        return embedding, metadata

    def encode_condition(self, retargeted_pkl: Path) -> tuple[np.ndarray, dict[str, Any]]:
        qpos, fps, source_metadata = load_rotation_condition(Path(retargeted_pkl))
        embedding, derived_metadata = self.encode_qpos(qpos, fps)
        return embedding, {"source": source_metadata, "derived": derived_metadata}

    def encode_motion_directory(self, motion_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
        qpos, fps, source_metadata = load_rotation_target(Path(motion_dir))
        embedding, derived_metadata = self.encode_qpos(qpos, fps)
        return embedding, {"source": source_metadata, "derived": derived_metadata}

    @staticmethod
    def mm_distance(condition_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
        condition = np.asarray(condition_embeddings, dtype=np.float32)
        motion = np.asarray(motion_embeddings, dtype=np.float32)
        if condition.shape != motion.shape:
            raise ValueError(f"Embedding shape mismatch: {condition.shape} vs {motion.shape}")
        return float(np.linalg.norm(condition - motion, axis=1).mean())

    @staticmethod
    def retrieval_at_k(
        motion_embeddings: np.ndarray,
        condition_embeddings: np.ndarray,
        top_k: int = 3,
    ) -> np.ndarray:
        motion = np.asarray(motion_embeddings, dtype=np.float32)
        condition = np.asarray(condition_embeddings, dtype=np.float32)
        if motion.shape != condition.shape or len(motion) != 32:
            raise ValueError("Strict OMG-compatible retrieval requires one aligned fixed batch of 32 pairs")
        distances = np.linalg.norm(motion[:, None, :] - condition[None, :, :], axis=-1)
        ranking = np.argsort(distances, axis=1)
        diagonal = np.arange(len(motion))[:, None]
        return np.asarray([(ranking[:, :k] == diagonal).any(axis=1).mean() for k in range(1, top_k + 1)])
