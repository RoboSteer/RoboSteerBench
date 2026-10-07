from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .model import RhythmMotionAlignment
from .omg_encoder import FrozenOMGMotionEncoder
from .preprocessing import load_rotation_target, qpos_to_rotation_features
from .runtime import make_windows, resample_qpos, sha256_file


class RhythmMotionEvaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        omg_root: Path,
        omg_checkpoint: Path,
        device: torch.device | str | None = None,
        ast_model_path: Path | None = None,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from transformers import AutoFeatureExtractor, ASTForAudioClassification

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.omg_root = Path(omg_root).resolve()
        self.omg_checkpoint = Path(omg_checkpoint).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "rhythm_motion_alignment" not in state:
            raise ValueError(f"Invalid Rhythm-Motion checkpoint: {self.checkpoint_path}")
        architecture = state.get("architecture") or {}
        expected_hash = str(architecture.get("omg_checkpoint_sha256") or "")
        actual_hash = sha256_file(self.omg_checkpoint)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError("OMG evaluator checkpoint SHA256 does not match the training checkpoint")

        self.ast_model_name = str(ast_model_path or architecture["audio_backbone"])
        self.ast_revision = str(architecture["audio_backbone_revision"])
        self.sample_rate = 16_000
        self.audio_chunk_samples = 10 * self.sample_rate
        self.audio_stride_samples = 5 * self.sample_rate
        self.derived_fps = 30
        self.motion_window_frames = 60
        self.motion_window_stride = 30
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.ast_model_name,
            revision=self.ast_revision,
        )
        self.ast_model = ASTForAudioClassification.from_pretrained(
            self.ast_model_name,
            revision=self.ast_revision,
        ).to(self.device)
        self.ast_model.requires_grad_(False).eval()
        self.model = RhythmMotionAlignment(
            int(architecture["audio_feature_dim"]),
            int(architecture["embedding_dim"]),
        ).to(self.device)
        self.model.load_state_dict(state["rhythm_motion_alignment"], strict=True)
        self.model.requires_grad_(False).eval()
        self.motion_encoder = FrozenOMGMotionEncoder(
            self.omg_root,
            self.omg_checkpoint,
            self.device,
            embedding_dim=int(architecture["embedding_dim"]),
        )
        self.checkpoint_metadata = {
            "architecture": architecture,
            "training": state.get("training"),
            "validation": state.get("validation"),
        }

    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        waveform, original_sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = np.asarray(waveform.mean(axis=1), dtype=np.float32)
        if len(waveform) == 0:
            raise ValueError(f"Empty audio: {audio_path}")
        if original_sr != self.sample_rate:
            target_samples = max(1, int(round(len(waveform) * self.sample_rate / original_sr)))
            waveform = F.interpolate(
                torch.from_numpy(waveform)[None, None],
                size=target_samples,
                mode="linear",
                align_corners=False,
            )[0, 0].numpy()
        return waveform, int(original_sr)

    @torch.inference_mode()
    def encode_audio_batch(
        self, audio_paths: list[Path], *, chunk_batch_size: int = 8
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Batch AST forwards over chunks while retaining per-audio pooling."""
        if chunk_batch_size <= 0:
            raise ValueError("chunk_batch_size must be positive")
        prepared: list[tuple[Path, np.ndarray, int, list[int], list[np.ndarray]]] = []
        chunk_rows: list[tuple[int, np.ndarray]] = []
        for owner, path in enumerate(audio_paths):
            audio_path = Path(path).resolve()
            waveform, original_sr = self._load_audio(audio_path)
            if len(waveform) <= self.audio_chunk_samples:
                starts = [0]
            else:
                starts = list(range(0, len(waveform) - self.audio_chunk_samples + 1, self.audio_stride_samples))
                tail = len(waveform) - self.audio_chunk_samples
                if starts[-1] != tail:
                    starts.append(tail)
            chunks = [waveform[start : start + self.audio_chunk_samples] for start in starts]
            prepared.append((audio_path, waveform, original_sr, starts, chunks))
            chunk_rows.extend((owner, chunk) for chunk in chunks)
        feature_dim = int(self.model.audio_projection.in_features)
        sums = np.zeros((len(prepared), feature_dim), dtype=np.float32)
        counts = np.zeros(len(prepared), dtype=np.int64)
        for start in range(0, len(chunk_rows), chunk_batch_size):
            group = chunk_rows[start : start + chunk_batch_size]
            inputs = self.feature_extractor(
                [chunk for _, chunk in group], sampling_rate=self.sample_rate,
                return_tensors="pt", padding="max_length", truncation=True,
            )
            outputs = self.ast_model(
                **{key: value.to(self.device) for key, value in inputs.items()},
                output_hidden_states=True, return_dict=True,
            )
            features = outputs.hidden_states[-1][:, :2].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            for row, (owner, _) in enumerate(group):
                sums[owner] += features[row]
                counts[owner] += 1
        if np.any(counts == 0):
            raise RuntimeError("rhythm batching produced an input without chunks")
        with torch.inference_mode():
            embeddings = self.model(torch.from_numpy(sums / counts).to(self.device)).cpu().numpy().astype(np.float32)
        output: list[tuple[np.ndarray, dict[str, Any]]] = []
        for index, (audio_path, waveform, original_sr, starts, _) in enumerate(prepared):
            output.append((embeddings[index], {
                "audio_path": str(audio_path),
                "original_sample_rate": original_sr,
                "derived_sample_rate": self.sample_rate,
                "derived_samples": int(len(waveform)),
                "duration_seconds": float(len(waveform) / self.sample_rate),
                "backbone": self.ast_model_name,
                "backbone_revision": self.ast_revision,
                "chunk_seconds": 10,
                "stride_seconds": 5,
                "chunk_starts_16khz": starts,
                "aggregation": "mean CLS/distillation tokens, then mean chunks",
            }))
        return output

    @torch.inference_mode()
    def encode_audio(self, audio_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        audio_path = Path(audio_path).resolve()
        waveform, original_sr = self._load_audio(audio_path)
        if len(waveform) <= self.audio_chunk_samples:
            starts = [0]
        else:
            starts = list(
                range(0, len(waveform) - self.audio_chunk_samples + 1, self.audio_stride_samples)
            )
            tail = len(waveform) - self.audio_chunk_samples
            if starts[-1] != tail:
                starts.append(tail)
        chunks = [waveform[start : start + self.audio_chunk_samples] for start in starts]
        inputs = self.feature_extractor(
            chunks,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        outputs = self.ast_model(
            **{key: value.to(self.device) for key, value in inputs.items()},
            output_hidden_states=True,
            return_dict=True,
        )
        ast_feature = outputs.hidden_states[-1][:, :2].mean(dim=1).mean(dim=0, keepdim=True)
        embedding = self.model(ast_feature)[0].cpu().numpy().astype(np.float32)
        metadata = {
            "audio_path": str(audio_path),
            "original_sample_rate": original_sr,
            "derived_sample_rate": self.sample_rate,
            "derived_samples": int(len(waveform)),
            "duration_seconds": float(len(waveform) / self.sample_rate),
            "backbone": self.ast_model_name,
            "backbone_revision": self.ast_revision,
            "chunk_seconds": 10,
            "stride_seconds": 5,
            "chunk_starts_16khz": starts,
            "aggregation": "mean CLS/distillation tokens, then mean chunks",
        }
        return embedding, metadata

    @torch.inference_mode()
    def encode_qpos(self, qpos_wxyz: np.ndarray, source_fps: float) -> tuple[np.ndarray, dict[str, Any]]:
        qpos = np.asarray(qpos_wxyz, dtype=np.float32)
        qpos_30fps, frame_times = resample_qpos(qpos, source_fps, self.derived_fps)
        windows, valid_masks, records = make_windows(
            qpos_30fps,
            self.motion_window_frames,
            self.motion_window_stride,
        )
        embedding = self.motion_encoder.encode_windows(windows, valid_masks)
        metadata = {
            "source_fps": float(source_fps),
            "source_frames": int(len(qpos)),
            "derived_fps": self.derived_fps,
            "derived_frames": int(len(qpos_30fps)),
            "derived_frame_times": frame_times.tolist(),
            "window_frames": self.motion_window_frames,
            "window_stride": self.motion_window_stride,
            "windows": [asdict(record) for record in records],
            "qpos_layout": "root_xyz + root_quaternion_wxyz + 29_joint_angles",
        }
        return embedding, metadata

    @torch.inference_mode()
    def encode_qpos_batch(
        self, qpos_batch: list[np.ndarray], source_fps: float, *, window_batch_size: int = 128
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]] = []
        windows_all: list[np.ndarray] = []
        masks_all: list[np.ndarray] = []
        owners: list[int] = []
        for owner, source in enumerate(qpos_batch):
            qpos = np.asarray(source, dtype=np.float32)
            if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
                raise ValueError(f"qpos_wxyz must have shape (T,36), T >= 2; got {qpos.shape}")
            qpos30, frame_times = resample_qpos(qpos, source_fps, self.derived_fps)
            windows, masks, records = make_windows(qpos30, self.motion_window_frames, self.motion_window_stride)
            prepared.append((qpos, qpos30, frame_times, records))
            windows_all.extend(windows)
            masks_all.extend(masks)
            owners.extend([owner] * len(windows))
        dim = int(self.checkpoint_metadata.get("architecture", {}).get("embedding_dim", 512))
        sums = np.zeros((len(prepared), dim), dtype=np.float32)
        counts = np.zeros(len(prepared), dtype=np.int64)
        for start in range(0, len(windows_all), window_batch_size):
            stop = min(start + window_batch_size, len(windows_all))
            embeddings = self.motion_encoder.encode_window_embeddings(
                np.stack(windows_all[start:stop]), np.stack(masks_all[start:stop])
            ).detach().cpu().numpy().astype(np.float32)
            for row, owner in enumerate(owners[start:stop]):
                sums[owner] += embeddings[row]
                counts[owner] += 1
        if np.any(counts == 0):
            raise RuntimeError("rhythm motion batching produced an input without windows")
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise RuntimeError("rhythm motion batching produced a zero-norm embedding")
        embeddings = sums / norms
        return [
            (embeddings[index].astype(np.float32), {
                "source_fps": float(source_fps),
                "source_frames": int(len(source)),
                "derived_fps": self.derived_fps,
                "derived_frames": int(len(qpos30)),
                "derived_frame_times": frame_times.tolist(),
                "window_frames": self.motion_window_frames,
                "window_stride": self.motion_window_stride,
                "windows": [asdict(record) for record in records],
                "qpos_layout": "root_xyz + root_quaternion_wxyz + 29_joint_angles",
            })
            for index, (source, qpos30, frame_times, records) in enumerate(prepared)
        ]

    def encode_motion_directory(self, motion_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
        qpos, fps, source_metadata = load_rotation_target(Path(motion_dir))
        embedding, derived_metadata = self.encode_qpos(qpos, fps)
        return embedding, {"source": source_metadata, "derived": derived_metadata}

    @staticmethod
    def mm_distance(audio_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
        audio = np.asarray(audio_embeddings, dtype=np.float32)
        motion = np.asarray(motion_embeddings, dtype=np.float32)
        if audio.shape != motion.shape:
            raise ValueError(f"Embedding shape mismatch: {audio.shape} vs {motion.shape}")
        return float(np.linalg.norm(audio - motion, axis=1).mean())

    @staticmethod
    def retrieval_at_k(
        motion_embeddings: np.ndarray,
        audio_embeddings: np.ndarray,
        top_k: int = 3,
    ) -> np.ndarray:
        motion = np.asarray(motion_embeddings, dtype=np.float32)
        audio = np.asarray(audio_embeddings, dtype=np.float32)
        if motion.shape != audio.shape or len(motion) != 32:
            raise ValueError("Strict OMG-compatible retrieval requires one aligned fixed batch of 32 pairs")
        distances = np.linalg.norm(motion[:, None, :] - audio[None, :, :], axis=-1)
        ranking = np.argsort(distances, axis=1)
        diagonal = np.arange(32)[:, None]
        return np.asarray([(ranking[:, :k] == diagonal).any(axis=1).mean() for k in range(1, top_k + 1)])
