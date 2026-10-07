"""Deterministic single- and multi-GPU scheduling for MM cache chunks."""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator
from scripts.evaluation.pipeline.mm_worker import initialize_mm_worker, worker_encode_chunk, worker_protocol

if TYPE_CHECKING:
    from scripts.evaluation.pipeline.context import EvaluationContext


def mm_gpu_devices(config: dict[str, Any], context_device: str) -> list[str]:
    raw = config.get("mm_gpu_devices")
    if raw is None or raw == "":
        return []
    if context_device == "cpu":
        raise ValueError("mm_gpu_devices cannot be used with --device cpu")
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    devices: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text.isdigit():
            text = f"cuda:{int(text)}"
        elif text.startswith("cuda:") and text[5:].isdigit():
            text = f"cuda:{int(text[5:])}"
        else:
            raise ValueError(
                "mm_gpu_devices entries must be GPU indices or cuda:N strings"
            )
        if text in devices:
            raise ValueError(f"duplicate MM GPU device: {text}")
        devices.append(text)
    if not devices:
        raise ValueError("mm_gpu_devices did not contain a valid GPU index")
    return devices


def mm_runtime_overrides(
    config: dict[str, Any],
    gpu_count: int,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    local_cache_root = str(config.get("video_local_cache_root", "")).strip()
    if local_cache_root:
        overrides["video_local_cache_root"] = local_cache_root
    if gpu_count > 1:
        decode_workers = int(config.get("mm_video_decode_workers_per_gpu", 6))
        if decode_workers <= 0:
            raise ValueError("mm_video_decode_workers_per_gpu must be positive")
        overrides["video_decode_workers"] = decode_workers
    xclip_batch = config.get("mm_xclip_window_batch_size")
    if xclip_batch is not None:
        xclip_batch = int(xclip_batch)
        if xclip_batch <= 0:
            raise ValueError("mm_xclip_window_batch_size must be positive")
        overrides["xclip_window_batch_size"] = xclip_batch
    return overrides


def encode_chunks_single_gpu(
    *,
    context: "EvaluationContext",
    evaluator: CrossModalMotionEvaluator,
    chunk_specs: list[tuple[int, list[str]]],
    cache_root: Path,
    source_fps: float,
    pair_batch_size: int,
    preprocess_workers: int,
    profile: str,
    total_pairs: int,
) -> tuple[dict[int, dict[str, Any]], int, int]:
    from scripts.evaluation.pipeline.mm_pipeline import _encode_chunk, _load_chunk, _save_chunk

    results: dict[int, dict[str, Any]] = {}
    cached_source_samples = 0
    computed_source_samples = 0
    with tqdm(
        total=total_pairs,
        desc=f"Encoding {profile} pairs",
        unit="pair",
        dynamic_ncols=True,
    ) as progress:
        for chunk_index, requested_ids in chunk_specs:
            chunk_path = cache_root / f"{chunk_index:06d}.npz"
            chunk = _load_chunk(chunk_path, requested_ids)
            if chunk is None:
                chunk = _encode_chunk(
                    requested_ids,
                    context.prediction_index.samples,
                    context.instruction_index,
                    evaluator,
                    source_fps,
                    pair_batch_size,
                    preprocess_workers,
                )
                _save_chunk(chunk_path, requested_ids, chunk)
                computed_source_samples += len(requested_ids)
            else:
                cached_source_samples += len(requested_ids)
            results[chunk_index] = chunk
            progress.update(len(requested_ids))
    return results, cached_source_samples, computed_source_samples


def encode_chunks_multi_gpu(
    *,
    context: "EvaluationContext",
    profile: str,
    common_ids: list[str],
    chunk_specs: list[tuple[int, list[str]]],
    source_fps: float,
    pair_batch_size: int,
    cache_chunk_size: int,
    preprocess_workers: int,
    gpu_devices: list[str],
    runtime_overrides: dict[str, Any],
) -> tuple[
    dict[str, Any],
    Path,
    dict[int, dict[str, Any]],
    int,
    int,
]:
    from scripts.evaluation.pipeline.mm_pipeline import _cache_root, _load_chunk, _save_chunk

    prediction_samples = {
        sample_id: context.prediction_index.samples[sample_id]
        for sample_id in common_ids
    }
    spawn_context = get_context("spawn")
    results: dict[int, dict[str, Any]] = {}
    cached_source_samples = 0
    computed_source_samples = 0

    with ExitStack() as stack:
        executors: list[ProcessPoolExecutor] = []
        for device in gpu_devices:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=spawn_context,
                initializer=initialize_mm_worker,
                initargs=(
                    profile,
                    context.models,
                    device,
                    runtime_overrides,
                    prediction_samples,
                    context.instruction_index,
                    source_fps,
                    pair_batch_size,
                    preprocess_workers,
                ),
            )
            executors.append(stack.enter_context(executor))

        # Loading/probing on worker zero avoids creating a competing evaluator
        # and CUDA context in the parent process.
        evaluator_protocol = executors[0].submit(worker_protocol).result()
        cache_root = _cache_root(
            context,
            profile,
            common_ids,
            source_fps,
            cache_chunk_size,
            evaluator_protocol,
        )

        missing: list[tuple[int, list[str]]] = []
        with tqdm(
            total=len(common_ids),
            desc=f"Encoding {profile} pairs on {len(gpu_devices)} GPUs",
            unit="pair",
            dynamic_ncols=True,
        ) as progress:
            for chunk_index, requested_ids in chunk_specs:
                chunk_path = cache_root / f"{chunk_index:06d}.npz"
                cached = _load_chunk(chunk_path, requested_ids)
                if cached is None:
                    missing.append((chunk_index, requested_ids))
                else:
                    results[chunk_index] = cached
                    cached_source_samples += len(requested_ids)
                    progress.update(len(requested_ids))

            jobs = iter(missing)
            pending: dict[
                Future[tuple[int, dict[str, Any]]],
                tuple[int, int, list[str]],
            ] = {}

            def submit_next(worker_index: int) -> None:
                try:
                    chunk_index, requested_ids = next(jobs)
                except StopIteration:
                    return
                future = executors[worker_index].submit(
                    worker_encode_chunk,
                    chunk_index,
                    requested_ids,
                )
                pending[future] = (worker_index, chunk_index, requested_ids)

            for worker_index in range(min(len(executors), len(missing))):
                submit_next(worker_index)

            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: pending[item][1]):
                    worker_index, expected_index, requested_ids = pending.pop(future)
                    try:
                        chunk_index, chunk = future.result()
                    except Exception as exc:
                        device = gpu_devices[worker_index]
                        raise RuntimeError(
                            f"MM worker on {device} failed for chunk {expected_index}"
                        ) from exc
                    if chunk_index != expected_index:
                        raise RuntimeError(
                            f"MM worker returned chunk {chunk_index}, expected {expected_index}"
                        )
                    chunk_path = cache_root / f"{chunk_index:06d}.npz"
                    _save_chunk(chunk_path, requested_ids, chunk)
                    results[chunk_index] = chunk
                    computed_source_samples += len(requested_ids)
                    progress.update(len(requested_ids))
                    submit_next(worker_index)

    return (
        evaluator_protocol,
        cache_root,
        results,
        cached_source_samples,
        computed_source_samples,
    )
