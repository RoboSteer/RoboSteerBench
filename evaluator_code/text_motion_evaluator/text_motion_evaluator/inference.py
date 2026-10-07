from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .model import TextMotionAlignment
from .omg_encoder import FrozenOMGMotionEncoder
from .runtime import load_qpos36, make_windows, resample_qpos, sha256_file


class TextMotionEvaluator:
    """Inference interface for the trained Text-Motion semantic evaluator."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        omg_root: Path,
        omg_checkpoint: Path,
        device: torch.device | str | None = None,
        model_cache: Path | None = None,
        text_model_path: Path | str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.omg_root = Path(omg_root).resolve()
        self.omg_checkpoint = Path(omg_checkpoint).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if not self.omg_checkpoint.is_file():
            raise FileNotFoundError(self.omg_checkpoint)

        omg_source = str(self.omg_root / "src")
        if omg_source not in sys.path:
            sys.path.insert(0, omg_source)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "text_motion_alignment" not in state:
            raise ValueError(f"Invalid Text-Motion checkpoint: {self.checkpoint_path}")
        architecture = state.get("architecture") or {}
        expected_omg_hash = architecture.get("omg_checkpoint_sha256")
        actual_omg_hash = sha256_file(self.omg_checkpoint)
        if expected_omg_hash and actual_omg_hash != expected_omg_hash:
            raise ValueError("OMG evaluator checkpoint SHA256 does not match the training checkpoint")

        self.embedding_dim = int(architecture.get("embedding_dim", 512))
        self.text_feature_dim = int(architecture.get("text_feature_dim", 768))
        self.text_model_name = str(Path(text_model_path).resolve() if text_model_path is not None else architecture.get("text_backbone", "google-t5/t5-base"))
        self.text_revision = str(
            architecture.get("text_backbone_revision", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1")
        )
        self.text_max_length = int(architecture.get("text_max_length", 100))
        self.derived_fps = 30
        self.window_frames = 60
        self.window_stride = 30

        self.alignment = TextMotionAlignment(self.text_feature_dim, self.embedding_dim).to(self.device)
        self.alignment.load_state_dict(state["text_motion_alignment"], strict=True)
        self.alignment.requires_grad_(False).eval()

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from transformers import AutoTokenizer, T5EncoderModel
        cache_dir = str(Path(model_cache).resolve()) if model_cache is not None else None
        load_kwargs = {"cache_dir": cache_dir}
        if text_model_path is None:
            load_kwargs["revision"] = self.text_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_name, use_fast=True, **load_kwargs
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.text_model_name, **load_kwargs
        ).to(self.device)
        self.text_encoder.requires_grad_(False).eval()
        if int(self.text_encoder.config.d_model) != self.text_feature_dim:
            raise ValueError(
                f"Text backbone dimension mismatch: {self.text_encoder.config.d_model} vs {self.text_feature_dim}"
            )
        self.motion_encoder = FrozenOMGMotionEncoder(
            self.omg_root,
            self.omg_checkpoint,
            self.device,
            embedding_dim=self.embedding_dim,
        )
        self.checkpoint_metadata = {
            "architecture": architecture,
            "validation": state.get("validation"),
            "training": state.get("training"),
        }

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        values = [str(text).strip() for text in texts]
        if not values or any(not text for text in values):
            raise ValueError("texts must contain at least one non-empty string")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        embeddings: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.text_encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            projected = self.alignment(pooled).cpu().numpy().astype(np.float32)
            embeddings.append(projected)
            for offset, text in enumerate(batch):
                valid_tokens = int(tokens["attention_mask"][offset].sum())
                metadata.append(
                    {
                        "text": text,
                        "backbone": self.text_model_name,
                        "backbone_revision": self.text_revision,
                        "backbone_frozen": True,
                        "pooling": "attention_mask_mean_last_hidden_state",
                        "max_length": self.text_max_length,
                        "valid_tokens": valid_tokens,
                        "truncated": valid_tokens == self.text_max_length,
                        "embedding_dim": self.embedding_dim,
                    }
                )
        return np.concatenate(embeddings, axis=0), metadata

    def encode_text(self, text: str) -> tuple[np.ndarray, dict[str, Any]]:
        embeddings, metadata = self.encode_texts([text], batch_size=1)
        return embeddings[0], metadata[0]

    @torch.inference_mode()
    def encode_qpos(self, qpos_wxyz: np.ndarray, source_fps: float) -> tuple[np.ndarray, dict[str, Any]]:
        qpos = np.asarray(qpos_wxyz, dtype=np.float32)
        qpos_30fps, frame_times = resample_qpos(qpos, source_fps, self.derived_fps)
        windows, valid_masks, window_records = make_windows(
            qpos_30fps, self.window_frames, self.window_stride
        )
        embedding = self.motion_encoder.encode_windows(windows, valid_masks)
        metadata = {
            "source_fps": float(source_fps),
            "source_frames": int(len(qpos)),
            "derived_fps": self.derived_fps,
            "derived_frames": int(len(qpos_30fps)),
            "derived_frame_times": frame_times.tolist(),
            "window_frames": self.window_frames,
            "window_stride": self.window_stride,
            "windows": [asdict(item) for item in window_records],
            "window_aggregation": "mean_then_l2_normalize",
            "qpos_layout": "root_xyz + root_quaternion_wxyz + 29_joint_angles",
            "embedding_dim": self.embedding_dim,
        }
        return embedding, metadata

    def encode_motion_directory(self, motion_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
        motion_dir = Path(motion_dir).resolve()
        qpos, source_metadata = load_qpos36(motion_dir)
        info = source_metadata["source_info"]
        source_fps = float(info.get("target_fps", 0))
        hold_frames = int(info.get("start_hold_frames", 0))
        if source_fps <= 0:
            raise ValueError(f"Missing target_fps in {motion_dir / 'info.txt'}")
        if hold_frames < 0 or hold_frames >= len(qpos):
            raise ValueError(f"Invalid start_hold_frames={hold_frames} for {len(qpos)} frames")
        embedding, derived_metadata = self.encode_qpos(qpos[hold_frames:], source_fps)
        return embedding, {
            "source": {
                **source_metadata,
                "removed_start_hold_frames": hold_frames,
                "frames_after_hold_removal": int(len(qpos) - hold_frames),
            },
            "derived": derived_metadata,
        }

    @staticmethod
    def mm_distance(text_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
        text = np.asarray(text_embeddings, dtype=np.float32)
        motion = np.asarray(motion_embeddings, dtype=np.float32)
        if text.shape != motion.shape or text.ndim != 2:
            raise ValueError(f"Embedding shape mismatch: {text.shape} vs {motion.shape}")
        return float(np.linalg.norm(text - motion, axis=1).mean())

    @staticmethod
    def retrieval_at_k(
        motion_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        top_k: int = 3,
    ) -> np.ndarray:
        motion = np.asarray(motion_embeddings, dtype=np.float64)
        text = np.asarray(text_embeddings, dtype=np.float64)
        if motion.shape != text.shape or motion.ndim != 2 or len(motion) != 32:
            raise ValueError("Strict OMG-compatible retrieval requires one aligned fixed batch of 32 pairs")
        distances = (
            -2.0 * (motion @ text.T)
            + np.square(motion).sum(axis=1, keepdims=True)
            + np.square(text).sum(axis=1)[None, :]
        )
        paired = np.diag(distances)
        ranks = (distances < paired[:, None]).sum(axis=1) + 1
        return np.asarray([(ranks <= k).mean() for k in range(1, top_k + 1)], dtype=np.float64)
