"""Shared batched and resumable execution for pluggable X--motion evaluators."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex
from scripts.evaluation.data.motion import load_qpos_36
from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator, load_cross_modal_evaluator
from scripts.evaluation.pipeline.mm_scheduler import (
    encode_chunks_multi_gpu as _encode_chunks_multi_gpu,
    encode_chunks_single_gpu as _encode_chunks_single_gpu,
    mm_gpu_devices as _mm_gpu_devices,
    mm_runtime_overrides as _mm_runtime_overrides,
)

if TYPE_CHECKING:
    from scripts.evaluation.data.instruction import InstructionSample
    from .context import EvaluationContext


@dataclass(frozen=True)
class _PairInput:
    sample_id: str
    qpos_36: np.ndarray
    condition: "InstructionSample"
    semantic_id: str
    dataset_name: str
    identity_source: str


def encode_mm_pairs(
    context: "EvaluationContext",
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Encode ID-aligned condition--prediction pairs in reusable mini-batches.

    CPU qpos loading is parallelised, GPU work is delegated to the evaluator's
    batch hooks, and completed groups are persisted under ``output/cache``.
    The cache is deterministic for an unchanged input manifest and can resume
    a stopped MM-Distance/R@K run without recomputing completed groups.
    """
    profile = str(context.config.get("mm_encoder", "")).strip()
    if not profile:
        raise ValueError(
            "MM-Distance requires --mm-encoder, for example --mm-encoder rhythm_motion"
        )
    if context.instruction_index is None:
        raise ValueError("MM-Distance requires --instruction-groundtruth")

    cache = getattr(context, "_cross_modal_mm_cache", None)
    if isinstance(cache, dict) and profile in cache:
        return cache[profile]

    common_ids = sorted(
        set(context.prediction_index.samples) & set(context.instruction_index.samples)
    )
    if not common_ids:
        raise RuntimeError("prediction and instruction groundtruth have no shared sample IDs")

    pair_batch_size = int(context.config.get("mm_pair_batch_size", 8))
    cache_chunk_size = int(context.config.get("mm_cache_chunk_size", 256))
    workers = int(context.config.get("preprocess_workers", 1))
    if pair_batch_size <= 0 or cache_chunk_size <= 0 or workers <= 0:
        raise ValueError("mm_pair_batch_size, mm_cache_chunk_size, and preprocess_workers must be positive")
    source_fps = float(context.config["prediction_fps"])
    gpu_devices = _mm_gpu_devices(context.config, context.device)
    runtime_overrides = _mm_runtime_overrides(context.config, len(gpu_devices))
    condition_aggregation = context.instruction_index.metadata.get("condition_aggregation")
    if condition_aggregation is not None:
        runtime_overrides["video_condition_aggregation"] = str(condition_aggregation)
    chunk_specs = list(enumerate(_chunks(common_ids, cache_chunk_size)))
    if len(gpu_devices) > 1:
        worker_preprocess = int(
            context.config.get(
                "mm_preprocess_workers_per_gpu",
                max(1, workers // len(gpu_devices)),
            )
        )
        if worker_preprocess <= 0:
            raise ValueError("mm_preprocess_workers_per_gpu must be positive")
        (
            evaluator_protocol,
            cache_root,
            chunk_results,
            cached_source_samples,
            computed_source_samples,
        ) = _encode_chunks_multi_gpu(
            context=context,
            profile=profile,
            common_ids=common_ids,
            chunk_specs=chunk_specs,
            source_fps=source_fps,
            pair_batch_size=pair_batch_size,
            cache_chunk_size=cache_chunk_size,
            preprocess_workers=worker_preprocess,
            gpu_devices=gpu_devices,
            runtime_overrides=runtime_overrides,
        )
        execution_workers = worker_preprocess
        pipeline_name = "multi_gpu_batched_resumable_x_motion_v2"
    else:
        evaluator_device = gpu_devices[0] if gpu_devices else context.device
        evaluator = load_cross_modal_evaluator(
            profile,
            context.models,
            evaluator_device,
            runtime_overrides=runtime_overrides,
        )
        evaluator_protocol = evaluator.protocol()
        cache_root = _cache_root(
            context,
            profile,
            common_ids,
            source_fps,
            cache_chunk_size,
            evaluator_protocol,
        )
        (
            chunk_results,
            cached_source_samples,
            computed_source_samples,
        ) = _encode_chunks_single_gpu(
            context=context,
            evaluator=evaluator,
            chunk_specs=chunk_specs,
            cache_root=cache_root,
            source_fps=source_fps,
            pair_batch_size=pair_batch_size,
            preprocess_workers=workers,
            profile=profile,
            total_pairs=len(common_ids),
        )
        execution_workers = workers
        pipeline_name = "single_gpu_batched_resumable_x_motion_v2"

    sample_ids: list[str] = []
    motion_values: list[np.ndarray] = []
    condition_values: list[np.ndarray] = []
    retrieval_ids: list[str] = []
    dataset_names: list[str] = []
    identity_sources: list[str] = []
    invalid: dict[str, str] = {}
    for chunk_index, _ in chunk_specs:
        _extend_from_chunk(
            chunk_results[chunk_index],
            sample_ids,
            motion_values,
            condition_values,
            retrieval_ids,
            dataset_names,
            identity_sources,
            invalid,
        )

    if not sample_ids:
        examples = list(invalid.items())[:3]
        raise RuntimeError(
            f"no valid pairs for cross-modal evaluator {profile!r}; examples: {examples}"
        )
    motion_embeddings = np.stack(motion_values).astype(np.float32)
    condition_embeddings = np.stack(condition_values).astype(np.float32)
    context.invalid[f"mm_distance_{profile}"] = invalid
    context._save_embeddings(f"mm_{profile}_motion", sample_ids, motion_embeddings)
    context._save_embeddings(f"mm_{profile}_condition", sample_ids, condition_embeddings)
    retrieval_metadata = {
        "semantic_ids": retrieval_ids,
        "dataset_names": dataset_names,
        "identity_sources": identity_sources,
        "identity_source": identity_sources[0] if len(set(identity_sources)) == 1 else "mixed",
    }
    protocol = {
        **evaluator_protocol,
        "execution": {
            "pipeline": pipeline_name,
            "pair_batch_size": pair_batch_size,
            "cache_chunk_size": cache_chunk_size,
            "preprocess_workers_per_process": execution_workers,
            "preprocess_workers_total": execution_workers * max(1, len(gpu_devices)),
            "worker_processes": max(1, len(gpu_devices)),
            "gpu_devices": gpu_devices or [context.device],
            "cache_root": str(cache_root),
            "video_local_cache_root": context.config.get("video_local_cache_root"),
            "cached_source_samples": cached_source_samples,
            "computed_source_samples": computed_source_samples,
        },
    }
    result = (sample_ids, motion_embeddings, condition_embeddings, protocol, retrieval_metadata)
    if not isinstance(cache, dict):
        cache = {}
        setattr(context, "_cross_modal_mm_cache", cache)
    cache[profile] = result
    return result


def _encode_chunk(
    requested_ids: list[str],
    prediction_samples: dict[str, Path],
    instruction_index: InstructionIndex,
    evaluator: CrossModalMotionEvaluator,
    source_fps: float,
    pair_batch_size: int,
    workers: int,
) -> dict[str, Any]:
    valid: list[_PairInput] = []
    invalid: dict[str, str] = {}

    def load_one(sample_id: str) -> _PairInput:
        condition = instruction_index.samples[sample_id]
        identity, dataset, identity_source = _retrieval_identity(condition, evaluator.name)
        qpos = load_qpos_36(prediction_samples[sample_id])
        return _PairInput(sample_id, qpos, condition, identity, dataset, identity_source)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mm-preprocess") as pool:
        futures = {sample_id: pool.submit(load_one, sample_id) for sample_id in requested_ids}
        for sample_id in requested_ids:
            try:
                valid.append(futures[sample_id].result())
            except Exception as exc:
                invalid[sample_id] = str(exc)

    sample_ids: list[str] = []
    motion_values: list[np.ndarray] = []
    condition_values: list[np.ndarray] = []
    semantic_ids: list[str] = []
    dataset_names: list[str] = []
    identity_sources: list[str] = []
    for batch in _chunks(valid, pair_batch_size):
        encoded, failed = _encode_batch(
            evaluator, batch, instruction_index, source_fps
        )
        invalid.update(failed)
        for pair, motion, condition in encoded:
            sample_ids.append(pair.sample_id)
            motion_values.append(motion)
            condition_values.append(condition)
            semantic_ids.append(pair.semantic_id)
            dataset_names.append(pair.dataset_name)
            identity_sources.append(pair.identity_source)
    return {
        "sample_ids": sample_ids,
        "motion_embeddings": _stack_or_empty(motion_values),
        "condition_embeddings": _stack_or_empty(condition_values),
        "semantic_ids": semantic_ids,
        "dataset_names": dataset_names,
        "identity_sources": identity_sources,
        "invalid": invalid,
    }


def _encode_batch(
    evaluator: CrossModalMotionEvaluator,
    batch: list[_PairInput],
    instruction_index: InstructionIndex,
    source_fps: float,
) -> tuple[list[tuple[_PairInput, np.ndarray, np.ndarray]], dict[str, str]]:
    try:
        motions = evaluator.encode_motion_batch([pair.qpos_36 for pair in batch], source_fps)
        conditions = evaluator.encode_condition_batch(
            [pair.condition for pair in batch], instruction_index
        )
        if len(motions) != len(batch) or len(conditions) != len(batch):
            raise RuntimeError(
                f"batch evaluator returned {len(motions)} motion and {len(conditions)} condition embeddings "
                f"for {len(batch)} inputs"
            )
        output = []
        for pair, motion_row, condition_row in zip(batch, motions, conditions, strict=True):
            motion = _embedding_vector(motion_row[0], pair.sample_id, "motion")
            condition = _embedding_vector(condition_row[0], pair.sample_id, "condition")
            if motion.shape != condition.shape:
                raise ValueError(
                    f"embedding shape mismatch: motion {motion.shape}, condition {condition.shape}"
                )
            output.append((pair, motion, condition))
        return output, {}
    except Exception:
        # Preserve the old per-sample fault isolation when a malformed input
        # prevents a native batched evaluator from processing its whole batch.
        output: list[tuple[_PairInput, np.ndarray, np.ndarray]] = []
        invalid: dict[str, str] = {}
        for pair in batch:
            try:
                motion, _ = evaluator.encode_motion(pair.qpos_36, source_fps)
                condition, _ = evaluator.encode_condition(
                    pair.condition, instruction_index
                )
                motion = _embedding_vector(motion, pair.sample_id, "motion")
                condition = _embedding_vector(condition, pair.sample_id, "condition")
                if motion.shape != condition.shape:
                    raise ValueError(
                        f"embedding shape mismatch: motion {motion.shape}, condition {condition.shape}"
                    )
                output.append((pair, motion, condition))
            except Exception as exc:
                invalid[pair.sample_id] = str(exc)
        return output, invalid


def _cache_root(
    context: "EvaluationContext", profile: str, sample_ids: list[str], source_fps: float,
    cache_chunk_size: int, evaluator_protocol: dict[str, Any],
) -> Path:
    payload = {
        "schema": "liujiahui.mm_pair_cache.v2",
        "profile": profile,
        "prediction": str(context.prediction_index.root.resolve()),
        "instruction": str(context.instruction_index.root.resolve()),  # type: ignore[union-attr]
        "source_fps": source_fps,
        "cache_chunk_size": cache_chunk_size,
        "evaluator_protocol": evaluator_protocol,
        "sample_ids": sample_ids,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    root = context.output / "cache" / f"mm_pairs_{profile}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_chunk(path: Path, requested_ids: list[str], chunk: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            requested_sample_ids=np.asarray(requested_ids, dtype=np.str_),
            sample_ids=np.asarray(chunk["sample_ids"], dtype=np.str_),
            motion_embeddings=np.asarray(chunk["motion_embeddings"], dtype=np.float32),
            condition_embeddings=np.asarray(chunk["condition_embeddings"], dtype=np.float32),
            semantic_ids=np.asarray(chunk["semantic_ids"], dtype=np.str_),
            dataset_names=np.asarray(chunk["dataset_names"], dtype=np.str_),
            identity_sources=np.asarray(chunk["identity_sources"], dtype=np.str_),
            invalid_json=np.asarray(json.dumps(chunk["invalid"], ensure_ascii=False)),
        )
    temporary.replace(path)


def _load_chunk(path: Path, requested_ids: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload["requested_sample_ids"].astype(str).tolist() != requested_ids:
                return None
            sample_ids = payload["sample_ids"].astype(str).tolist()
            motion = np.asarray(payload["motion_embeddings"], dtype=np.float32)
            condition = np.asarray(payload["condition_embeddings"], dtype=np.float32)
            semantic_ids = payload["semantic_ids"].astype(str).tolist()
            dataset_names = payload["dataset_names"].astype(str).tolist()
            identity_sources = payload["identity_sources"].astype(str).tolist()
            invalid = json.loads(str(payload["invalid_json"].item()))
        count = len(sample_ids)
        if not (
            motion.ndim == 2 and condition.shape == motion.shape and
            len(semantic_ids) == len(dataset_names) == len(identity_sources) == count and
            np.isfinite(motion).all() and np.isfinite(condition).all()
        ):
            return None
        return {
            "sample_ids": sample_ids,
            "motion_embeddings": motion,
            "condition_embeddings": condition,
            "semantic_ids": semantic_ids,
            "dataset_names": dataset_names,
            "identity_sources": identity_sources,
            "invalid": {str(key): str(value) for key, value in invalid.items()},
        }
    except Exception:
        return None


def _extend_from_chunk(
    chunk: dict[str, Any], sample_ids: list[str], motion_values: list[np.ndarray],
    condition_values: list[np.ndarray], semantic_ids: list[str], dataset_names: list[str],
    identity_sources: list[str], invalid: dict[str, str],
) -> None:
    ids = list(chunk["sample_ids"])
    motion = np.asarray(chunk["motion_embeddings"], dtype=np.float32)
    condition = np.asarray(chunk["condition_embeddings"], dtype=np.float32)
    if len(ids) != len(motion) or len(ids) != len(condition):
        raise RuntimeError("MM pair cache has inconsistent embedding rows")
    sample_ids.extend(ids)
    motion_values.extend(motion)
    condition_values.extend(condition)
    semantic_ids.extend(chunk["semantic_ids"])
    dataset_names.extend(chunk["dataset_names"])
    identity_sources.extend(chunk["identity_sources"])
    invalid.update(chunk["invalid"])


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stack_or_empty(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(values).astype(np.float32)


def _embedding_vector(value: np.ndarray, sample_id: str, side: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]
    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError(
            f"{side} encoder for {sample_id!r} must return (D,), got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{side} encoder for {sample_id!r} returned non-finite values")
    return embedding


def _retrieval_identity(sample: Any, profile: str) -> tuple[str, str, str]:
    """Resolve the explicit retrieval identity, with a documented MUL fallback."""
    attributes = getattr(sample, "attributes", {}) or {}
    for key in ("semantic_id", "kinematic_id", "retrieval_id", "condition_id"):
        value = attributes.get(key)
        if value is not None and str(value):
            dataset = str(attributes.get("dataset") or attributes.get("condition") or profile)
            return str(value), dataset, f"manifest.{key}"
    sample_id = str(sample.sample_id)
    normalized_id = sample_id.lstrip("_")
    prefix = normalized_id.split("_", 1)[0]
    if not prefix:
        raise ValueError(f"cannot derive retrieval identity from sample_id {sample_id!r}")
    source = (
        "sample_id.leading_underscores_stripped_before_first_underscore"
        if normalized_id != sample_id else "sample_id.prefix_before_first_underscore"
    )
    return prefix, str(attributes.get("dataset") or profile), source
