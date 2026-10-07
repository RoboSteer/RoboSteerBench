"""Spawn-safe worker state for multi-GPU cross-modal encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.evaluation.data.instruction import InstructionIndex
from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator, load_cross_modal_evaluator


@dataclass
class _WorkerState:
    evaluator: CrossModalMotionEvaluator
    prediction_samples: dict[str, Path]
    instruction_index: InstructionIndex
    source_fps: float
    pair_batch_size: int
    preprocess_workers: int


_STATE: _WorkerState | None = None


def initialize_mm_worker(
    profile: str,
    models: Path,
    device: str,
    runtime_overrides: dict[str, Any],
    prediction_samples: dict[str, Path],
    instruction_index: InstructionIndex,
    source_fps: float,
    pair_batch_size: int,
    preprocess_workers: int,
) -> None:
    """Load one evaluator on one explicit device for the worker lifetime."""
    global _STATE
    evaluator = load_cross_modal_evaluator(
        profile,
        models,
        device,
        runtime_overrides=runtime_overrides,
    )
    _STATE = _WorkerState(
        evaluator=evaluator,
        prediction_samples=prediction_samples,
        instruction_index=instruction_index,
        source_fps=source_fps,
        pair_batch_size=pair_batch_size,
        preprocess_workers=preprocess_workers,
    )


def worker_protocol() -> dict[str, Any]:
    return _require_state().evaluator.protocol()


def worker_encode_chunk(
    chunk_index: int,
    requested_ids: list[str],
) -> tuple[int, dict[str, Any]]:
    # Import lazily to avoid an mm_pipeline <-> mm_worker import cycle.
    from scripts.evaluation.pipeline.mm_pipeline import _encode_chunk

    state = _require_state()
    chunk = _encode_chunk(
        requested_ids,
        state.prediction_samples,
        state.instruction_index,
        state.evaluator,
        state.source_fps,
        state.pair_batch_size,
        state.preprocess_workers,
    )
    return chunk_index, chunk


def _require_state() -> _WorkerState:
    if _STATE is None:
        raise RuntimeError("MM worker was not initialized")
    return _STATE
