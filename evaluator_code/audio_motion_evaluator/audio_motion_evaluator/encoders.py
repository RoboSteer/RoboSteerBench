from __future__ import annotations

import sys
import time
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ProjectConfig
from .io_utils import sha256_file


def _add_omg_to_path(config: ProjectConfig) -> None:
    source = str((config.omg_root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def _motion_encoder_kwargs(config: dict[str, Any], input_dim: int, output_dim: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_dim": input_dim, "output_dim": output_dim}
    for key in (
        "movement_dim",
        "hidden_dim",
        "movement_mode",
        "temporal_kind",
        "num_layers",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "max_len",
    ):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


class FrozenOMGMotionEncoder:
    def __init__(self, config: ProjectConfig, device: torch.device):
        _add_omg_to_path(config)
        from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
        from omg.benchmarks.evaluator.representation import motion_input_dim
        from omg.robots.g1.kinematics import G1Kinematics

        print(f"[motion encoder] loading {config.omg_checkpoint}", flush=True)
        started = time.perf_counter()
        checkpoint = torch.load(
            str(config.omg_checkpoint), map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(checkpoint, dict) or "motion_encoder" not in checkpoint:
            raise ValueError(f"Invalid OMG evaluator checkpoint: {config.omg_checkpoint}")
        ckpt_config = checkpoint.get("config") or {}
        motion_key = str(ckpt_config.get("motion_key", "body_pos_local"))
        if motion_key != "body_pos_local":
            raise ValueError(f"V1 requires OMG motion_key='body_pos_local', checkpoint has {motion_key!r}")
        relative_kinematics = Path(str(ckpt_config.get("kinematics_path", "assets/robots/g1/g1_kinematics.json")))
        kinematics_path = relative_kinematics if relative_kinematics.is_absolute() else config.omg_root / relative_kinematics
        input_dim = motion_input_dim(motion_key, kinematics_path=str(kinematics_path))
        output_dim = int(ckpt_config.get("embedding_dim", config.embedding_dim))
        self.encoder = MotionEncoder(**_motion_encoder_kwargs(ckpt_config, input_dim, output_dim)).to(device)
        self.encoder.load_state_dict(checkpoint["motion_encoder"], strict=True)
        self.encoder.requires_grad_(False).eval()
        self.kinematics = G1Kinematics(kinematics_path=str(kinematics_path))
        self.device = device
        self.config = ckpt_config
        self.checkpoint_logit_scale = checkpoint.get("logit_scale")
        print("[motion encoder] hashing checkpoint once for provenance", flush=True)
        self.checkpoint_sha256 = sha256_file(config.omg_checkpoint)
        print(f"[motion encoder] ready in {time.perf_counter() - started:.1f}s", flush=True)

    @torch.inference_mode()
    def encode_window_embeddings(self, qpos_windows: np.ndarray, valid_masks: np.ndarray) -> torch.Tensor:
        """Encode independent motion windows without source-level aggregation."""
        from omg.benchmarks.evaluator.representation import canonical_body_positions_from_qpos

        qpos = torch.as_tensor(qpos_windows, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(valid_masks, dtype=torch.bool, device=self.device)
        body_pos_local = canonical_body_positions_from_qpos(qpos, self.kinematics)
        return self.encoder(body_pos_local, valid_mask=valid)

    @torch.inference_mode()
    def encode_windows(self, qpos_windows: np.ndarray, valid_masks: np.ndarray) -> np.ndarray:
        """Backward-compatible one-source aggregation over its windows."""
        window_embeddings = self.encode_window_embeddings(qpos_windows, valid_masks)
        pair_embedding = F.normalize(window_embeddings.mean(dim=0, keepdim=True), dim=-1)
        return pair_embedding[0].cpu().numpy().astype(np.float32)


class FrozenWhisperSpeechEncoder:
    """Frozen Whisper encoder with traceable full-audio chunk aggregation."""

    def __init__(self, config: ProjectConfig, device: torch.device):
        # hf-xet has been unreliable in the target Windows environment; standard
        # HTTP keeps the official Hugging Face cache format and supports retries.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from transformers import WhisperFeatureExtractor, WhisperModel

        print(f"[speech encoder] loading {config.whisper_model}", flush=True)
        started = time.perf_counter()
        config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.whisper_model,
            revision=config.whisper_revision,
            cache_dir=config.model_cache_dir,
            # The packaged snapshot is complete; evaluation must not retry network access.
            local_files_only=True,
        )
        self.model = WhisperModel.from_pretrained(
            config.whisper_model,
            revision=config.whisper_revision,
            cache_dir=config.model_cache_dir,
            local_files_only=True,
        ).to(device)
        self.model.requires_grad_(False).eval()
        self.device = device
        self.sample_rate = config.audio_sample_rate
        self.hidden_dim = int(self.model.config.d_model)
        self.model_name = config.whisper_model
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.whisper_revision
        print(f"[speech encoder] ready in {time.perf_counter() - started:.1f}s", flush=True)

    def _prepare_file(self, audio_path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Decode one file and construct its deterministic full-audio chunks."""
        audio_path = Path(audio_path).resolve()
        waveform, original_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = np.asarray(waveform.mean(axis=1), dtype=np.float32)
        original_samples = int(len(waveform))
        if waveform.size == 0:
            raise ValueError(f"empty audio file: {audio_path}")
        if original_sr != self.sample_rate:
            target_samples = max(1, int(round(len(waveform) * self.sample_rate / original_sr)))
            tensor = torch.from_numpy(waveform)[None, None, :]
            waveform = F.interpolate(
                tensor, size=target_samples, mode="linear", align_corners=False
            )[0, 0].numpy()
        chunk_samples = 30 * self.sample_rate
        stride_samples = 25 * self.sample_rate
        if len(waveform) <= chunk_samples:
            starts = [0]
        else:
            starts = list(range(0, len(waveform) - chunk_samples + 1, stride_samples))
            tail = len(waveform) - chunk_samples
            if starts[-1] != tail:
                starts.append(tail)
        chunks = [waveform[start : start + chunk_samples] for start in starts]
        metadata = {
            "audio_path": str(audio_path),
            "original_sample_rate": int(original_sr),
            "original_samples": original_samples,
            "derived_sample_rate": self.sample_rate,
            "derived_samples": int(len(waveform)),
            "duration_seconds": float(len(waveform) / self.sample_rate),
            "resampling": "torch_linear_interpolation_align_corners_false" if original_sr != self.sample_rate else "none",
            "chunk_policy": {
                "chunk_seconds": 30,
                "stride_seconds": 25,
                "tail_policy": "append_final_full_chunk",
                "aggregation": "masked_mean_hidden_then_mean_chunks",
            },
            "chunks": [
                {
                    "index": index,
                    "start_sample_16khz": start,
                    "end_sample_exclusive_16khz": min(start + chunk_samples, len(waveform)),
                    "valid_hidden_frames": min(
                        1500, max(1, int(np.ceil(len(chunk) / 160 / 2)))
                    ),
                }
                for index, (start, chunk) in enumerate(zip(starts, chunks, strict=True))
            ],
            "backbone": self.model_name,
            "backbone_revision": self.revision,
            "hidden_dim": self.hidden_dim,
        }
        return chunks, metadata

    @torch.inference_mode()
    def encode_files(
        self, audio_paths: list[Path], *, chunk_batch_size: int = 8
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Batch Whisper forward passes while preserving per-file pooling exactly."""
        if chunk_batch_size <= 0:
            raise ValueError("chunk_batch_size must be positive")
        prepared = [self._prepare_file(path) for path in audio_paths]
        chunk_rows: list[tuple[int, np.ndarray, int]] = []
        for owner, (chunks, _) in enumerate(prepared):
            for chunk in chunks:
                valid_hidden_frames = min(1500, max(1, int(np.ceil(len(chunk) / 160 / 2))))
                chunk_rows.append((owner, chunk, valid_hidden_frames))
        sums = np.zeros((len(prepared), self.hidden_dim), dtype=np.float32)
        counts = np.zeros(len(prepared), dtype=np.int64)
        for start in range(0, len(chunk_rows), chunk_batch_size):
            group = chunk_rows[start : start + chunk_batch_size]
            inputs = self.feature_extractor(
                [chunk for _, chunk, _ in group], sampling_rate=self.sample_rate,
                return_tensors="pt", padding="max_length", truncation=True,
            )
            hidden = self.model.encoder(
                input_features=inputs.input_features.to(self.device)
            ).last_hidden_state
            pooled = np.stack([
                hidden[row, :valid_frames].mean(dim=0).detach().cpu().numpy()
                for row, (_, _, valid_frames) in enumerate(group)
            ]).astype(np.float32)
            for row, (owner, _, _) in enumerate(group):
                sums[owner] += pooled[row]
                counts[owner] += 1
        if np.any(counts == 0):
            raise RuntimeError("audio batching produced an input without chunks")
        return [
            ((sums[index] / counts[index]).astype(np.float32, copy=False), metadata)
            for index, (_, metadata) in enumerate(prepared)
        ]

    @torch.inference_mode()
    def encode_file(self, audio_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Backward-compatible single-file call."""
        return self.encode_files([audio_path], chunk_batch_size=1)[0]


class AudioProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.normalization(self.projection(audio_features)), dim=-1)
