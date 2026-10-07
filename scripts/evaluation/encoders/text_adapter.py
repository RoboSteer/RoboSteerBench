"""Adapter for the delivered OMG Text-Motion semantic evaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex, InstructionSample
from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator, _resolve_path
from scripts.evaluation.omg_paths import omg_root as default_omg_root


class TextMotionAdapter(CrossModalMotionEvaluator):
    """Bridge the delivered TextMotionEvaluator to the benchmark API."""

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        super().__init__(name, config, registry_root, device)
        delivery_root = _resolve_path(config["root"], registry_root)
        checkpoint = _resolve_path(config["checkpoint"], registry_root)
        omg_root = _resolve_path(config["omg_root"], registry_root)
        omg_checkpoint = _resolve_path(
            config.get("omg_checkpoint", default_omg_root() / "models/evaluator/step_004000.pt"),
            registry_root,
        )
        model_cache_value = config.get("model_cache")
        model_cache = _resolve_path(model_cache_value, registry_root) if model_cache_value else None
        text_model_path = _resolve_path(
            config.get("text_model_path", config["root"]), registry_root
        )
        for label, path, is_dir in (
            ("text evaluator root", delivery_root, True),
            ("text evaluator checkpoint", checkpoint, False),
            ("OMG root", omg_root, True),
            ("OMG motion checkpoint", omg_checkpoint, False),
            ("T5 backbone", text_model_path, True),
        ):
            if not (path.is_dir() if is_dir else path.is_file()):
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(delivery_root) not in sys.path:
            sys.path.insert(0, str(delivery_root))
        try:
            from text_motion_evaluator.inference import TextMotionEvaluator
        except ImportError as exc:
            raise RuntimeError(
                "text_motion requires the delivered text_motion_evaluator package"
            ) from exc

        evaluator_device = None if device == "auto" else device
        self._evaluator = TextMotionEvaluator(
            checkpoint,
            omg_root=omg_root,
            omg_checkpoint=omg_checkpoint,
            device=evaluator_device,
            model_cache=model_cache,
            text_model_path=text_model_path,
        )
        self._paths = {
            "checkpoint": str(checkpoint),
            "omg_checkpoint": str(omg_checkpoint),
            "model_cache": None if model_cache is None else str(model_cache),
        }

    @staticmethod
    def _text(sample: InstructionSample) -> str:
        text = (getattr(sample, "attributes", {}) or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"text_motion requires a non-empty manifest 'text' field for {sample.sample_id!r}"
            )
        return text.strip()

    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del index
        embedding, metadata = self._evaluator.encode_text(self._text(sample))
        return np.asarray(embedding, dtype=np.float32), {
            "sample_id": sample.sample_id,
            **metadata,
        }

    def encode_condition_batch(
        self, samples: list[InstructionSample], index: InstructionIndex
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        del index
        texts = [self._text(sample) for sample in samples]
        values, metadata = self._evaluator.encode_texts(
            texts, batch_size=int(self.config.get("text_batch_size", 32))
        )
        return [
            (
                values[index].astype(np.float32, copy=False),
                {"sample_id": sample.sample_id, **metadata[index]},
            )
            for index, sample in enumerate(samples)
        ]

    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        embedding, metadata = self._evaluator.encode_qpos(qpos_36, source_fps)
        return np.asarray(embedding, dtype=np.float32), metadata

    def encode_motion_batch(
        self, qpos_36_batch: list[np.ndarray], source_fps: float
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        # The delivered evaluator exposes a single-sequence qpos API. The
        # shared MM pipeline still batches file loading and text encoding.
        return [self.encode_motion(qpos, source_fps) for qpos in qpos_36_batch]

    def protocol(self) -> dict[str, Any]:
        return {
            "evaluator": "TextMotionEvaluator",
            "condition": "manifest_inline_text",
            "motion_input": "qpos_36_resampled_and_windowed_by_delivery_evaluator",
            "window_frames": int(self._evaluator.window_frames),
            "window_stride": int(self._evaluator.window_stride),
            "derived_fps": int(self._evaluator.derived_fps),
            "embedding_dim": int(self._evaluator.embedding_dim),
            "text_backbone": self._evaluator.text_model_name,
            "text_backbone_revision": self._evaluator.text_revision,
            "text_max_length": int(self._evaluator.text_max_length),
            "l2_normalized": True,
            **self._paths,
        }
