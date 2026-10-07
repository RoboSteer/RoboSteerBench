"""Adapter for the delivered Whisper--OMG audio--motion evaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex, InstructionSample
from scripts.evaluation.encoders.cross_modal import (
    CrossModalMotionEvaluator,
    _resolve_condition_asset,
    _resolve_path,
)


class AudioMotionAdapter(CrossModalMotionEvaluator):
    """Encode complete raw audio and qpos motions in the supplied 512-D space."""

    _AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        super().__init__(name, config, registry_root, device)
        delivery_root = _resolve_path(config["root"], registry_root)
        checkpoint = _resolve_path(config["checkpoint"], registry_root)
        omg_root = _resolve_path(config["omg_root"], registry_root)
        omg_checkpoint = _resolve_path(config["omg_checkpoint"], registry_root)
        whisper_model_path = (
            _resolve_path(config["whisper_model_path"], registry_root)
            if config.get("whisper_model_path") else None
        )
        for label, path, is_dir in (
            ("audio evaluator root", delivery_root, True),
            ("audio evaluator checkpoint", checkpoint, False),
            ("OMG root", omg_root, True),
            ("OMG motion checkpoint", omg_checkpoint, False),
            *([("Whisper backbone", whisper_model_path, True)] if whisper_model_path else []),
        ):
            if not (path.is_dir() if is_dir else path.is_file()):
                raise FileNotFoundError(f"{label} does not exist: {path}")

        if str(delivery_root) not in sys.path:
            sys.path.insert(0, str(delivery_root))
        try:
            import torch
            from audio_motion_evaluator import AudioMotionEvaluator
        except ImportError as exc:
            raise RuntimeError(
                "audio_motion requires the dependencies in evaluator_code/audio_motion_evaluator/requirements.txt"
            ) from exc

        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for audio_motion but torch.cuda.is_available() is false")

        self._evaluator = AudioMotionEvaluator(
            checkpoint,
            omg_root=omg_root,
            omg_checkpoint=omg_checkpoint,
            device=torch_device,
            whisper_model_path=whisper_model_path,
        )
        self._condition_asset = str(config.get("condition_asset", "raw_audio"))
        self._paths = {
            "checkpoint": str(checkpoint),
            "omg_root": str(omg_root),
            "omg_checkpoint": str(omg_checkpoint),
        }

    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        audio_path = _resolve_condition_asset(
            sample,
            index,
            asset_name=self._condition_asset,
            suffixes=self._AUDIO_SUFFIXES,
            default_name_suffix="_audio",
        )
        embedding, metadata = self._evaluator.encode_audio(audio_path)
        return embedding, {"source_path": str(audio_path), **metadata}

    def encode_condition_batch(
        self, samples: list[InstructionSample], index: InstructionIndex
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        paths = [
            _resolve_condition_asset(
                sample, index, asset_name=self._condition_asset,
                suffixes=self._AUDIO_SUFFIXES, default_name_suffix="_audio",
            )
            for sample in samples
        ]
        values = self._evaluator.encode_audio_batch(
            paths, chunk_batch_size=int(self.config.get("audio_chunk_batch_size", 8))
        )
        return [
            (embedding, {"source_path": str(path), **metadata})
            for path, (embedding, metadata) in zip(paths, values, strict=True)
        ]

    def encode_motion_batch(
        self, qpos_36_batch: list[np.ndarray], source_fps: float
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        return self._evaluator.encode_motion_batch(
            qpos_36_batch, source_fps,
            window_batch_size=int(self.config.get("motion_window_batch_size", 128)),
        )

    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._evaluator.encode_motion(qpos_36, source_fps)

    def protocol(self) -> dict[str, Any]:
        architecture = self._evaluator.architecture
        config = self._evaluator.config
        return {
            "evaluator": "AudioMotionEvaluator",
            "condition": "complete_raw_audio",
            "condition_asset": self._condition_asset,
            "audio_backbone": str(architecture["speech_backbone"]),
            "audio_backbone_revision": str(architecture["speech_backbone_revision"]),
            "audio_sample_rate": int(config.audio_sample_rate),
            "audio_chunk_seconds": 30,
            "audio_stride_seconds": 25,
            "motion_input": "complete_qpos_36",
            "motion_target_fps": int(config.derived_fps),
            "motion_window_frames": int(config.window_frames),
            "motion_window_stride": int(config.window_stride),
            "motion_aggregation": "mean_window_embeddings_then_l2_normalize",
            "embedding_dim": int(architecture["embedding_dim"]),
            "l2_normalized": True,
            "evaluator_checkpoint_sha256": self._evaluator.checkpoint_sha256,
            **self._paths,
        }
