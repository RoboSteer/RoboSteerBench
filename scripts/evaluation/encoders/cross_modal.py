"""Pluggable cross-modal evaluators for paired X--motion embeddings."""

from __future__ import annotations

import importlib
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex, InstructionSample


class CrossModalMotionEvaluator(ABC):
    """Interface for two encoders trained in one shared embedding space."""

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        self.name = name
        self.config = config
        self.registry_root = registry_root
        self.device = device

    @abstractmethod
    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode one instruction/condition sample to a one-dimensional embedding."""

    @abstractmethod
    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode one complete qpos motion to a one-dimensional embedding."""

    def encode_condition_batch(
        self, samples: list[InstructionSample], index: InstructionIndex
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Encode a condition batch.

        The default preserves legacy evaluator behaviour exactly.  Evaluators
        with a native batch implementation override this method; the shared
        MM pipeline does not need modality-specific branches.
        """
        return [self.encode_condition(sample, index) for sample in samples]

    def encode_motion_batch(
        self, qpos_36_batch: list[np.ndarray], source_fps: float
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Encode a motion batch; subclasses may override for GPU batching."""
        return [self.encode_motion(qpos_36, source_fps) for qpos_36 in qpos_36_batch]

    @abstractmethod
    def protocol(self) -> dict[str, Any]:
        """Return the reproducible model and preprocessing protocol."""


class RhythmMotionAdapter(CrossModalMotionEvaluator):
    """Adapter around the delivered paired RhythmMotionEvaluator."""

    _AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        super().__init__(name, config, registry_root, device)
        delivery_root = _resolve_path(config["root"], registry_root)
        checkpoint = _resolve_path(config["checkpoint"], registry_root)
        omg_root = _resolve_path(config["omg_root"], registry_root)
        omg_checkpoint = _resolve_path(config["omg_checkpoint"], registry_root)
        ast_model_path = (
            _resolve_path(config["ast_model_path"], registry_root)
            if config.get("ast_model_path") else None
        )
        hf_home = _resolve_path(config.get("hf_home", "rhythm_encoder/hf_cache"), registry_root)
        for label, path, is_dir in (
            ("rhythm evaluator root", delivery_root, True),
            ("rhythm evaluator checkpoint", checkpoint, False),
            ("OMG root", omg_root, True),
            ("OMG motion checkpoint", omg_checkpoint, False),
            *([("AST backbone", ast_model_path, True)] if ast_model_path else []),
        ):
            if not (path.is_dir() if is_dir else path.is_file()):
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(delivery_root) not in sys.path:
            sys.path.insert(0, str(delivery_root))
        if ast_model_path is None:
            hf_home.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(hf_home))
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from rhythm_motion_evaluator import RhythmMotionEvaluator

        evaluator_device = None if device == "auto" else device
        self._evaluator = RhythmMotionEvaluator(
            checkpoint,
            omg_root=omg_root,
            omg_checkpoint=omg_checkpoint,
            device=evaluator_device,
            ast_model_path=ast_model_path,
        )
        self._condition_asset = str(config.get("condition_asset", "raw_audio"))
        self._paths = {
            "checkpoint": str(checkpoint),
            "omg_checkpoint": str(omg_checkpoint),
        }
        if ast_model_path is None:
            self._paths["hf_home"] = str(hf_home)
        else:
            self._paths["ast_model_path"] = str(ast_model_path)

    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        audio_path = _resolve_condition_asset(
            sample,
            index,
            asset_name=self._condition_asset,
            suffixes=self._AUDIO_SUFFIXES,
            default_name_suffix="_music",
        )
        return self._evaluator.encode_audio(audio_path)

    def encode_condition_batch(
        self, samples: list[InstructionSample], index: InstructionIndex
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        paths = [
            _resolve_condition_asset(
                sample, index, asset_name=self._condition_asset,
                suffixes=self._AUDIO_SUFFIXES, default_name_suffix="_music",
            )
            for sample in samples
        ]
        values = self._evaluator.encode_audio_batch(
            paths, chunk_batch_size=int(self.config.get("audio_chunk_batch_size", 8))
        )
        return list(values)

    def encode_motion_batch(
        self, qpos_36_batch: list[np.ndarray], source_fps: float
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        return self._evaluator.encode_qpos_batch(
            qpos_36_batch, source_fps,
            window_batch_size=int(self.config.get("motion_window_batch_size", 128)),
        )

    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._evaluator.encode_qpos(qpos_36, source_fps)

    def protocol(self) -> dict[str, Any]:
        architecture = self._evaluator.checkpoint_metadata.get("architecture") or {}
        return {
            "evaluator": "RhythmMotionEvaluator",
            "condition": "complete_raw_audio",
            "condition_asset": self._condition_asset,
            "audio_sample_rate": int(self._evaluator.sample_rate),
            "audio_chunk_seconds": float(
                self._evaluator.audio_chunk_samples / self._evaluator.sample_rate
            ),
            "audio_stride_seconds": float(
                self._evaluator.audio_stride_samples / self._evaluator.sample_rate
            ),
            "motion_input": "complete_qpos_36",
            "motion_target_fps": int(self._evaluator.derived_fps),
            "motion_window_frames": int(self._evaluator.motion_window_frames),
            "motion_window_stride": int(self._evaluator.motion_window_stride),
            "motion_aggregation": "mean_window_embeddings_then_l2_normalize",
            "embedding_dim": int(architecture.get("embedding_dim", 512)),
            "l2_normalized": True,
            **self._paths,
        }


def load_cross_modal_evaluator(
    name: str,
    registry_path: Path,
    device: str,
    runtime_overrides: dict[str, Any] | None = None,
) -> CrossModalMotionEvaluator:
    """Instantiate a registered evaluator through its importable factory."""
    registry_path = registry_path.resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"encoder registry does not exist: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    profiles = payload.get("cross_modal_evaluators")
    if not isinstance(profiles, dict):
        raise ValueError(f"{registry_path} does not define 'cross_modal_evaluators'")
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"cross-modal evaluator {name!r} is not registered; available: {available}")
    config = profiles[name]
    if not isinstance(config, dict):
        raise ValueError(f"cross-modal evaluator configuration for {name!r} must be an object")
    config = dict(config)
    if runtime_overrides:
        # Runtime-only execution knobs (for example a node-local byte cache)
        # must not require editing the model registry.
        config.update(runtime_overrides)
    external_omg = os.environ.get("ROBOOSTEER_OMG_ROOT")
    if external_omg:
        root = Path(external_omg).expanduser().resolve()
        config["omg_root"] = str(root)
        config["omg_checkpoint"] = str(root / "models/evaluator/step_004000.pt")
    factory_spec = str(config.get("factory", "")).strip()
    if not factory_spec or ":" not in factory_spec:
        raise ValueError(
            f"cross-modal evaluator {name!r} must declare factory as 'module:ClassName'"
        )
    module_name, symbol_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol_name)
    evaluator = factory(name, config, registry_path.parent, device)
    if not isinstance(evaluator, CrossModalMotionEvaluator):
        raise TypeError(f"factory {factory_spec!r} did not return CrossModalMotionEvaluator")
    return evaluator


def _resolve_condition_asset(
    sample: InstructionSample,
    index: InstructionIndex,
    *,
    asset_name: str,
    suffixes: tuple[str, ...],
    default_name_suffix: str,
) -> Path:
    assets = (getattr(sample, "attributes", {}) or {}).get("assets", {}) or {}
    if not assets:
        for row in index.metadata.get("samples", []):
            if str(row.get("sample_id")) == sample.sample_id:
                assets = row.get("assets", {}) or {}
                break
    if asset_name in assets:
        path = Path(str(assets[asset_name]))
        if not path.is_absolute():
            path = index.root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"condition asset does not exist: {path}")
        return path
    if sample.path.suffix.lower() in suffixes and sample.path.is_file():
        return sample.path
    raw_root_value = index.metadata.get("raw_root")
    raw_root = Path(raw_root_value).resolve() if raw_root_value else index.root / "raw"
    for stem in (sample.sample_id + default_name_suffix, sample.sample_id):
        for suffix in suffixes:
            candidate = raw_root / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"no {asset_name!r} asset for {sample.sample_id!r}; add samples[].assets.{asset_name} "
        f"to the instruction manifest or place the raw file under {raw_root}"
    )


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()
