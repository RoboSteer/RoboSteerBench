from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .model import VideoMotionAlignment
from .omg_encoder import FrozenOMGMotionEncoder
from .preprocessing import load_rotation_target
from .runtime import make_windows, resample_qpos, sha256_file
from .video import FrozenXCLIPVideoEncoder, inspect_video, sample_video_for_motion_windows


class VideoMotionEvaluator:
    """Inference API shared by the Human and Skeleton Video-Motion evaluators."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        omg_root: Path,
        omg_checkpoint: Path,
        model_cache_dir: Path | None = None,
        xclip_model_path: Path | None = None,
        expected_video_kind: str | None = None,
        device: torch.device | str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.omg_root = Path(omg_root).resolve()
        self.omg_checkpoint = Path(omg_checkpoint).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "video_motion_alignment" not in state:
            raise ValueError(f"Invalid Video-Motion checkpoint: {self.checkpoint_path}")
        self.video_kind = str(state.get("video_kind", ""))
        if self.video_kind not in {"human", "skeleton"}:
            raise ValueError(f"Invalid video_kind in checkpoint: {self.video_kind!r}")
        if expected_video_kind is not None and self.video_kind != expected_video_kind:
            raise ValueError(
                f"Expected {expected_video_kind!r} checkpoint, got {self.video_kind!r}"
            )
        architecture = state.get("architecture") or {}
        expected_omg_hash = architecture.get("omg_checkpoint_sha256")
        actual_omg_hash = sha256_file(self.omg_checkpoint)
        if expected_omg_hash and actual_omg_hash != expected_omg_hash:
            raise ValueError("OMG evaluator checkpoint SHA256 does not match the training checkpoint")

        self.derived_fps = 30
        self.window_frames = 60
        self.window_stride = 30
        self.frames_per_window = 8
        embedding_dim = int(architecture.get("embedding_dim", 512))
        video_dim = int(architecture.get("video_feature_dim", 512))
        self.alignment = VideoMotionAlignment(video_dim, embedding_dim).to(self.device)
        self.alignment.load_state_dict(state["video_motion_alignment"], strict=True)
        self.alignment.requires_grad_(False).eval()
        self.motion_encoder = FrozenOMGMotionEncoder(
            self.omg_root,
            self.omg_checkpoint,
            self.device,
            embedding_dim=embedding_dim,
        )
        video_config = SimpleNamespace(
            xclip_model=str(xclip_model_path or architecture.get("video_backbone", "microsoft/xclip-base-patch32")),
            xclip_revision=architecture.get(
                "video_backbone_revision",
                "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
            ),
            model_cache_dir=Path(model_cache_dir or self.checkpoint_path.parents[1] / "model_cache"),
        )
        self.video_encoder = FrozenXCLIPVideoEncoder(video_config, self.device)
        self.checkpoint_metadata = {
            "video_kind": self.video_kind,
            "architecture": architecture,
            "validation": state.get("validation"),
            "training": state.get("training"),
        }

    def _window_records_for_frames(self, frames_30fps: int) -> tuple[np.ndarray, list[Any]]:
        if frames_30fps <= 0:
            raise ValueError("frames_30fps must be positive")
        dummy = np.zeros((frames_30fps, 1), dtype=np.float32)
        _, valid_masks, window_records = make_windows(
            dummy,
            self.window_frames,
            self.window_stride,
        )
        return valid_masks, window_records

    @torch.inference_mode()
    def _project_video_windows(self, sampled_windows: np.ndarray) -> np.ndarray:
        frozen_feature = self.video_encoder.encode_windows(sampled_windows, batch_size=8)
        embedding = self.alignment(
            torch.from_numpy(frozen_feature).to(self.device)[None]
        )[0]
        return embedding.cpu().numpy().astype(np.float32)

    def _prepare_video(
        self,
        video_path: Path,
        *,
        reference_frames_30fps: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Decode and sample one video without invoking the XCLIP model.

        Keeping decoding separate lets ``encode_video_batch`` overlap the
        CPU/FFmpeg work for several files, then run their sampled windows
        through XCLIP together on the GPU.
        """
        video_path = Path(video_path)
        source = inspect_video(video_path)
        if reference_frames_30fps is None:
            last_time = (source["source_frames"] - 1) / source["source_fps"]
            reference_frames_30fps = max(1, int(round(last_time * self.derived_fps)) + 1)
            duration_policy = "derived from source video duration"
        else:
            reference_frames_30fps = int(reference_frames_30fps)
            duration_policy = "provided paired reference_frames_30fps"
        valid_masks, window_records = self._window_records_for_frames(reference_frames_30fps)
        sampled, sampling = sample_video_for_motion_windows(
            video_path,
            window_records,
            derived_fps=self.derived_fps,
            frames_per_window=self.frames_per_window,
            inspected_metadata=source,
        )
        metadata = {
            "video_kind": self.video_kind,
            "source": sampling,
            "derived": {
                "duration_policy": duration_policy,
                "reference_frames_30fps": reference_frames_30fps,
                "fps": self.derived_fps,
                "window_frames": self.window_frames,
                "window_stride": self.window_stride,
                "window_valid_masks": valid_masks.tolist(),
                "windows": [asdict(item) for item in window_records],
            },
        }
        return sampled, metadata

    def encode_video(
        self,
        video_path: Path,
        *,
        reference_frames_30fps: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode a condition video using its duration or a paired reference duration."""
        sampled, metadata = self._prepare_video(
            video_path, reference_frames_30fps=reference_frames_30fps
        )
        return self._project_video_windows(sampled), metadata

    @torch.inference_mode()
    def encode_video_batch(
        self,
        video_paths: list[Path],
        *,
        reference_frames_30fps: list[int | None] | None = None,
        decode_workers: int = 8,
        xclip_window_batch_size: int = 32,
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Encode videos with concurrent FFmpeg sampling and batched XCLIP.

        Window features are still L2-normalized, averaged per source video,
        normalized again, and passed through the same alignment network as in
        ``encode_video``. The method only changes execution parallelism.
        """
        paths = [Path(path) for path in video_paths]
        if not paths:
            return []
        references = reference_frames_30fps or [None] * len(paths)
        if len(references) != len(paths):
            raise ValueError("reference_frames_30fps must match video_paths")

        def prepare(index: int) -> tuple[np.ndarray, dict[str, Any]]:
            return self._prepare_video(paths[index], reference_frames_30fps=references[index])

        workers = max(1, min(int(decode_workers), len(paths)))
        if workers == 1:
            prepared = [prepare(index) for index in range(len(paths))]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-decode") as pool:
                prepared = list(pool.map(prepare, range(len(paths))))

        window_counts = [len(sampled) for sampled, _ in prepared]
        if any(count <= 0 for count in window_counts):
            raise ValueError("video sampling produced no windows")
        all_windows = [window for sampled, _ in prepared for window in sampled]
        window_features = self.video_encoder.encode_window_features(
            all_windows, batch_size=xclip_window_batch_size
        )

        pooled: list[torch.Tensor] = []
        start = 0
        for count in window_counts:
            per_video = torch.from_numpy(window_features[start:start + count])
            pooled.append(torch.nn.functional.normalize(per_video.mean(dim=0, keepdim=True), dim=-1))
            start += count
        embeddings = self.alignment(torch.cat(pooled, dim=0).to(self.device))
        return [
            (embedding.cpu().numpy().astype(np.float32), metadata)
            for embedding, (_, metadata) in zip(embeddings, prepared, strict=True)
        ]

    def _prepare_qpos(
        self,
        qpos_wxyz: np.ndarray,
        source_fps: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        qpos = np.asarray(qpos_wxyz, dtype=np.float32)
        qpos_30fps, frame_times = resample_qpos(qpos, source_fps, self.derived_fps)
        qpos_windows, valid_masks, window_records = make_windows(
            qpos_30fps,
            self.window_frames,
            self.window_stride,
        )
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
        return qpos_windows, valid_masks, metadata

    def encode_qpos(
        self,
        qpos_wxyz: np.ndarray,
        source_fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        qpos_windows, valid_masks, metadata = self._prepare_qpos(qpos_wxyz, source_fps)
        embedding = self.motion_encoder.encode_windows(qpos_windows, valid_masks)
        return embedding, metadata

    @torch.inference_mode()
    def encode_qpos_batch(
        self,
        qpos_wxyz_batch: list[np.ndarray],
        source_fps: float,
        *,
        window_batch_size: int = 2048,
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Batch OMG motion windows, preserving per-motion temporal pooling."""
        if not qpos_wxyz_batch:
            return []
        prepared = [self._prepare_qpos(qpos, source_fps) for qpos in qpos_wxyz_batch]
        counts = [len(qpos_windows) for qpos_windows, _, _ in prepared]
        windows = np.concatenate([qpos_windows for qpos_windows, _, _ in prepared], axis=0)
        masks = np.concatenate([valid_masks for _, valid_masks, _ in prepared], axis=0)
        window_embeddings = self.motion_encoder.encode_window_embeddings(
            windows, masks, batch_size=window_batch_size
        )
        output: list[tuple[np.ndarray, dict[str, Any]]] = []
        start = 0
        for count, (_, _, metadata) in zip(counts, prepared, strict=True):
            embedding = torch.nn.functional.normalize(
                torch.from_numpy(window_embeddings[start:start + count]).mean(dim=0, keepdim=True),
                dim=-1,
            )[0].numpy().astype(np.float32)
            output.append((embedding, metadata))
            start += count
        return output

    def encode_motion_directory(self, motion_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
        qpos, fps, source_metadata = load_rotation_target(Path(motion_dir))
        embedding, derived_metadata = self.encode_qpos(qpos, fps)
        return embedding, {"source": source_metadata, "derived": derived_metadata}

    @staticmethod
    def mm_distance(video_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
        video = np.asarray(video_embeddings, dtype=np.float64)
        motion = np.asarray(motion_embeddings, dtype=np.float64)
        if video.shape != motion.shape or video.ndim != 2:
            raise ValueError("Expected paired 2D embedding arrays with identical shapes")
        return float(np.linalg.norm(video - motion, axis=1).mean())

    @staticmethod
    def retrieval_at_k(
        motion_embeddings: np.ndarray,
        video_embeddings: np.ndarray,
        top_k: int = 3,
    ) -> np.ndarray:
        motion = np.asarray(motion_embeddings, dtype=np.float64)
        video = np.asarray(video_embeddings, dtype=np.float64)
        if motion.shape != video.shape or motion.ndim != 2 or len(motion) != 32:
            raise ValueError("Strict OMG-compatible retrieval requires one aligned fixed batch of 32 pairs")
        distances = (
            -2.0 * (motion @ video.T)
            + np.square(motion).sum(axis=1, keepdims=True)
            + np.square(video).sum(axis=1)[None, :]
        )
        paired = np.diag(distances)
        ranks = (distances < paired[:, None]).sum(axis=1) + 1
        return np.asarray([(ranks <= k).mean() for k in range(1, top_k + 1)], dtype=np.float64)
