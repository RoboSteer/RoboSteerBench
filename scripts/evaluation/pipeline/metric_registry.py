"""Metric names, dependencies, and execution adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

from scripts.evaluation.core_metric.BAS_Gap import bas_gap
from scripts.evaluation.core_metric.BehaviorGenerationScore import compute_behavior_generation_score
from scripts.evaluation.core_metric.BehaviorSteer_level2 import (
    amplitude_ir_2,
    body_restrain_ir_2,
    direction_ir_2,
    speed_ir_2,
)
from scripts.evaluation.core_metric.Trajectory import trajectory_ir_2
from scripts.evaluation.core_metric.Diversity import full_pair_diversity_gpu
from scripts.evaluation.core_metric.E_vel import VelocityErrorEvaluator, VelocityErrorScore
from scripts.evaluation.core_metric.FID import motion_fid
from scripts.evaluation.core_metric.g_MPJPE import (
    GlobalMPJPEEvaluator,
    GlobalMPJPEScore,
    OMG_SONIC_BODY_INDICES,
    groundtruth_start_hold_frames,
)
from scripts.evaluation.core_metric.MM_Distance import paired_mm_distance
from scripts.evaluation.core_metric.MPJPE import MPJPEScore, RootAlignedMPJPEEvaluator
from scripts.evaluation.core_metric.R_At_1 import fixed_batch_retrieval
from scripts.evaluation.data.motion import load_qpos_36
from scripts.evaluation.preprocessing.temporal import resample_qpos_to_frame_count

from .context import EvaluationContext
from .mm_pipeline import encode_mm_pairs


PREDICTION_MOTION = "prediction_motion"
MOTION_GROUNDTRUTH = "motion_groundtruth"
INSTRUCTION_GROUNDTRUTH = "instruction_groundtruth"
TEMPORAL_PRETRIMMED_GROUNDTRUTH_ROOTS = frozenset((
    "motion_fore",
    "motion_retro",
    "motion_inter",
))


@dataclass(frozen=True)
class MetricOutput:
    value: float
    num_samples: int
    details: dict[str, Any]
    sample_ids: list[str] | None = None
    sample_values: np.ndarray | None = None


@dataclass(frozen=True)
class MetricSpec:
    name: str
    requires: frozenset[str]
    direction: str
    run: Callable[[EvaluationContext], MetricOutput]


def _window_details(context: EvaluationContext, sample_ids: list[str]) -> dict[str, Any]:
    return {
        "unit": "motion_window",
        "num_source_samples": context.source_count(sample_ids),
        **context.window_protocol(),
    }


def _fid(context: EvaluationContext) -> MetricOutput:
    sample_ids, generated, reference = context.aligned(
        context.prediction_phase_embeddings(), context.groundtruth_phase_embeddings(),
        label="phase-normalized prediction and motion groundtruth",
    )
    return MetricOutput(
        motion_fid(reference, generated), len(sample_ids),
        {
            "embedding_dim": int(generated.shape[1]),
            "unit": "complete_source_motion",
            "num_source_samples": len(sample_ids),
            **context.phase_embedding_protocol(),
        },
    )


def _diversity(context: EvaluationContext) -> MetricOutput:
    """Exact all-unordered-pairs Diversity over one phase60 embedding per motion."""
    mapping = context.prediction_phase_embeddings()
    sample_ids = sorted(mapping)
    values = np.stack([mapping[key] for key in sample_ids])
    block_size = int(context.config.get("diversity_block_size", 4096))
    value = full_pair_diversity_gpu(
        values, device=context.device, block_size=block_size
    )
    pair_count = len(sample_ids) * (len(sample_ids) - 1) // 2
    return MetricOutput(value, len(sample_ids), {
        "embedding_dim": int(values.shape[1]),
        "unit": "complete_source_motion",
        "num_source_samples": len(sample_ids),
        "num_pairs": pair_count,
        "estimator": "exact_mean_over_all_unordered_embedding_pairs",
        "execution": "gpu_block_streaming_pairwise_l2",
        "block_size": block_size,
        "pair_constraint": "all_distinct_unordered_embedding_index_pairs",
        **context.phase_embedding_protocol(),
    })


def _contact_sliding(context: EvaluationContext) -> MetricOutput:
    mapping = context.prediction_contact_sliding()
    sample_ids = sorted(mapping)
    values = np.asarray([mapping[key].value for key in sample_ids], dtype=np.float64)
    contact_intervals = sum(mapping[key].num_contact_intervals for key in sample_ids)
    valid_intervals = sum(mapping[key].num_valid_intervals for key in sample_ids)
    return MetricOutput(float(values.mean()), len(sample_ids), {
        "unit": "m/s",
        "aggregation": "macro_mean_over_complete_source_motions",
        "num_contact_intervals": int(contact_intervals),
        "num_valid_foot_intervals": int(valid_intervals),
        "contact_interval_ratio": 0.0 if valid_intervals == 0 else float(contact_intervals / valid_intervals),
        **context.contact_sliding_protocol(),
    }, sample_ids, values)


def _matched_motion_ids(context: EvaluationContext, metric_name: str) -> list[str]:
    if context.motion_groundtruth_index is None:
        raise ValueError(f"{metric_name} requires --motion-groundtruth")
    sample_ids = sorted(set(context.prediction_index.samples) & set(context.motion_groundtruth_index.samples))
    if not sample_ids:
        raise RuntimeError(f"no matched prediction/GT pairs for {metric_name}")
    return sample_ids


def _physical_groundtruth_hold_frames(context: EvaluationContext, groundtruth_clip) -> int:
    """Return the GT initial hold used by the physical tracking metrics.

    The temporal-completion ground-truth roots ``motion_fore``,
    ``motion_retro``, and ``motion_inter`` store already-cropped target
    segments. Unlike ``motion_generation``, they have no simulator
    initial-hold prefix and do not carry generation-only ``info.txt`` files.
    All other motion roots retain the formal OMG Video/SONIC ``info.txt``
    contract.
    """
    if context.motion_groundtruth_index is None:
        raise ValueError("physical metrics require --motion-groundtruth")
    if context.motion_groundtruth_index.root.name in TEMPORAL_PRETRIMMED_GROUNDTRUTH_ROOTS:
        return 0
    return groundtruth_start_hold_frames(groundtruth_clip)


def _physical_groundtruth_time_policy(context: EvaluationContext) -> tuple[str, str, str]:
    """Return provenance strings matching the selected GT storage protocol."""
    if context.motion_groundtruth_index is None:
        raise ValueError("physical metrics require --motion-groundtruth")
    groundtruth_root_name = context.motion_groundtruth_index.root.name
    if groundtruth_root_name in TEMPORAL_PRETRIMMED_GROUNDTRUTH_ROOTS:
        return (
            f"{groundtruth_root_name}_pretrimmed_hold_zero_then_resample_prediction_to_effective_groundtruth_frames_if_needed_then_strict_equal_native_50hz_frames",
            "observed pretrimmed temporal GT CSV frames at fixed 50 FPS; no info.txt hold removal",
            f"{groundtruth_root_name}_pretrimmed_hold_zero",
        )
    return (
        "remove_groundtruth_info_start_hold_frames_then_resample_prediction_to_effective_groundtruth_frames_if_needed_then_strict_equal_native_50hz_frames",
        "observed CSV frames at fixed 50 FPS after info.txt start_hold_frames removal",
        "info.txt:start_hold_frames",
    )


def _physical_metric_qpos_pair(
    context: EvaluationContext, sample_id: str
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Load one physical-metric pair and resample only its prediction if needed.

    The prediction remains unchanged everywhere else in the evaluation. Its
    temporary metric-local copy is endpoint-preserving resampled only when its
    frame count differs from the GT frame count after the GT hold policy.
    """
    if context.motion_groundtruth_index is None:
        raise ValueError("physical metrics require --motion-groundtruth")
    groundtruth_clip = context.motion_groundtruth_index.samples[sample_id]
    prediction = load_qpos_36(context.prediction_index.samples[sample_id])
    groundtruth = load_qpos_36(groundtruth_clip)
    hold_frames = _physical_groundtruth_hold_frames(context, groundtruth_clip)
    effective_groundtruth_frames = len(groundtruth) - hold_frames
    if effective_groundtruth_frames < 2:
        raise ValueError(
            "groundtruth hold removal leaves fewer than two frames: "
            f"frames={len(groundtruth)}, hold_frames={hold_frames}"
        )
    prediction_resampled = len(prediction) != effective_groundtruth_frames
    if prediction_resampled:
        prediction = resample_qpos_to_frame_count(prediction, effective_groundtruth_frames)
    return prediction, groundtruth, hold_frames, prediction_resampled


def _mpjpe(context: EvaluationContext) -> MetricOutput:
    """Compute OMG Video/SONIC pelvis-translation-aligned MPJPE at native 50 Hz."""
    prediction_fps = float(context.config["prediction_fps"])
    groundtruth_fps = float(context.config["motion_groundtruth_fps"])
    physical_fps = float(context.config.get("mpjpe_fps", context.config.get("g_mpjpe_fps", 50)))
    evaluator = RootAlignedMPJPEEvaluator(device=context.device)
    time_policy, groundtruth_time_source, groundtruth_hold_source = _physical_groundtruth_time_policy(context)
    scores: dict[str, MPJPEScore] = {}
    invalid: dict[str, str] = {}
    prediction_resampled_ids: set[str] = set()
    for sample_id in tqdm(_matched_motion_ids(context, "MPJPE"), desc="Computing MPJPE", unit="motion", dynamic_ncols=True):
        try:
            prediction_qpos, groundtruth_qpos, hold_frames, prediction_resampled = _physical_metric_qpos_pair(
                context, sample_id
            )
            if prediction_resampled:
                prediction_resampled_ids.add(sample_id)
            scores[sample_id] = evaluator.score(
                prediction_qpos,
                groundtruth_qpos,
                prediction_fps=prediction_fps,
                groundtruth_fps=groundtruth_fps,
                target_fps=physical_fps,
                groundtruth_hold_frames=hold_frames,
            )
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not scores:
        raise RuntimeError("no valid prediction/GT pairs for MPJPE")
    context.invalid["mpjpe"] = invalid
    valid_ids = sorted(scores)
    values = np.asarray([scores[key].value for key in valid_ids], dtype=np.float64)
    frames = np.asarray([scores[key].num_frames for key in valid_ids], dtype=np.int32)
    durations = np.asarray([scores[key].duration_seconds for key in valid_ids], dtype=np.float32)
    np.savez_compressed(
        context.output / "cache" / "mpjpe.npz",
        sample_ids=np.asarray(valid_ids, dtype=np.str_),
        values_mm=values.astype(np.float32),
        num_frames=frames,
        duration_seconds=durations,
        fps=np.full(len(valid_ids), physical_fps, dtype=np.float32),
        groundtruth_hold_frames=np.asarray([scores[key].groundtruth_hold_frames for key in valid_ids], dtype=np.int32),
        prediction_resampled_to_groundtruth_frames=np.asarray(
            [key in prediction_resampled_ids for key in valid_ids], dtype=np.bool_
        ),
        time_policy=np.asarray(time_policy),
    )
    return MetricOutput(float(values.mean()), len(valid_ids), {
        "unit": "mm",
        "definition": "OMG Video/SONIC MPJPE: fixed-14-link FK L2 error after per-frame pelvis translation alignment only",
        "aggregation": "macro_mean_over_complete_matched_motions",
        "time_alignment": time_policy,
        "prediction_time_source": "observed CSV frames at fixed 50 FPS; resampled only within this physical metric when frame counts differ",
        "prediction_resampling": "endpoint-preserving linear translation/joints plus shortest-path quaternion SLERP to effective GT frame count, physical metrics only",
        "num_prediction_resampled_to_groundtruth_frames": len(prediction_resampled_ids),
        "groundtruth_time_source": groundtruth_time_source,
        "groundtruth_hold_source": groundtruth_hold_source,
        "prediction_fps": prediction_fps,
        "motion_groundtruth_fps": groundtruth_fps,
        "physical_fps": physical_fps,
        "position_set": "OMG Video/SONIC fixed 14 G1 FK body-link origins, including pelvis root",
        "body_indices": OMG_SONIC_BODY_INDICES.tolist(),
        "root_alignment": "per_frame_translation_only",
        "root_link": evaluator.root_link_name,
        "root_link_index": evaluator.root_link_index,
        "global_rotation_alignment": False,
        "procrustes_alignment": False,
        "num_joints": int(next(iter(scores.values())).num_joints),
        "mean_evaluated_frames": float(frames.mean()),
        "mean_groundtruth_hold_frames": float(np.mean([scores[key].groundtruth_hold_frames for key in valid_ids])),
        "root_error_after_alignment": 0.0,
        "no_temporal_resampling": False,
        "no_common_prefix_truncation": True,
        "unequal_pairs": "prediction_resampled_to_effective_groundtruth_frame_count_before_scoring",
    }, valid_ids, values)


def _g_mpjpe(context: EvaluationContext) -> MetricOutput:
    prediction_fps = float(context.config["prediction_fps"])
    groundtruth_fps = float(context.config["motion_groundtruth_fps"])
    physical_fps = float(context.config.get("g_mpjpe_fps", 50))
    evaluator = GlobalMPJPEEvaluator(device=context.device)
    time_policy, groundtruth_time_source, groundtruth_hold_source = _physical_groundtruth_time_policy(context)
    scores: dict[str, GlobalMPJPEScore] = {}
    invalid: dict[str, str] = {}
    prediction_resampled_ids: set[str] = set()
    for sample_id in tqdm(_matched_motion_ids(context, "g-MPJPE"), desc="Computing g-MPJPE", unit="motion", dynamic_ncols=True):
        try:
            prediction_qpos, groundtruth_qpos, hold_frames, prediction_resampled = _physical_metric_qpos_pair(
                context, sample_id
            )
            if prediction_resampled:
                prediction_resampled_ids.add(sample_id)
            scores[sample_id] = evaluator.score(
                prediction_qpos,
                groundtruth_qpos,
                prediction_fps=prediction_fps,
                groundtruth_fps=groundtruth_fps,
                target_fps=physical_fps,
                groundtruth_hold_frames=hold_frames,
            )
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not scores:
        raise RuntimeError("no valid prediction/GT pairs for g-MPJPE")
    context.invalid["g_mpjpe"] = invalid
    valid_ids = sorted(scores)
    values = np.asarray([scores[key].value for key in valid_ids], dtype=np.float64)
    frames = np.asarray([scores[key].num_frames for key in valid_ids], dtype=np.int32)
    durations = np.asarray([scores[key].duration_seconds for key in valid_ids], dtype=np.float32)
    np.savez_compressed(
        context.output / "cache" / "g_mpjpe.npz",
        sample_ids=np.asarray(valid_ids, dtype=np.str_),
        values_mm=values.astype(np.float32),
        num_frames=frames,
        duration_seconds=durations,
        fps=np.full(len(valid_ids), physical_fps, dtype=np.float32),
        groundtruth_hold_frames=np.asarray([scores[key].groundtruth_hold_frames for key in valid_ids], dtype=np.int32),
        prediction_resampled_to_groundtruth_frames=np.asarray(
            [key in prediction_resampled_ids for key in valid_ids], dtype=np.bool_
        ),
        time_policy=np.asarray(time_policy),
    )
    return MetricOutput(float(values.mean()), len(valid_ids), {
        "unit": "mm",
        "definition": "OMG Video/SONIC g-MPJPE: fixed-14-link world-coordinate L2 body-link position error without root alignment",
        "aggregation": "macro_mean_over_complete_matched_motions",
        "time_alignment": time_policy,
        "prediction_time_source": "observed CSV frames at fixed 50 FPS; resampled only within this physical metric when frame counts differ",
        "prediction_resampling": "endpoint-preserving linear translation/joints plus shortest-path quaternion SLERP to effective GT frame count, physical metrics only",
        "num_prediction_resampled_to_groundtruth_frames": len(prediction_resampled_ids),
        "groundtruth_time_source": groundtruth_time_source,
        "groundtruth_hold_source": groundtruth_hold_source,
        "prediction_fps": prediction_fps,
        "motion_groundtruth_fps": groundtruth_fps,
        "physical_fps": physical_fps,
        "position_set": "OMG Video/SONIC fixed 14 G1 FK body-link origins, including pelvis root",
        "body_indices": OMG_SONIC_BODY_INDICES.tolist(),
        "num_joints": int(next(iter(scores.values())).num_joints),
        "mean_evaluated_frames": float(frames.mean()),
        "mean_groundtruth_hold_frames": float(np.mean([scores[key].groundtruth_hold_frames for key in valid_ids])),
        "no_root_alignment": True,
        "no_procrustes_alignment": True,
        "no_temporal_resampling": False,
        "no_common_prefix_truncation": True,
        "unequal_pairs": "prediction_resampled_to_effective_groundtruth_frame_count_before_scoring",
    }, valid_ids, values)


def _e_vel(context: EvaluationContext) -> MetricOutput:
    """Compute OMG Video/SONIC E_vel at native 50 Hz."""
    prediction_fps = float(context.config["prediction_fps"])
    groundtruth_fps = float(context.config["motion_groundtruth_fps"])
    physical_fps = float(context.config.get("e_vel_fps", 50))
    evaluator = VelocityErrorEvaluator(device=context.device)
    time_policy, groundtruth_time_source, groundtruth_hold_source = _physical_groundtruth_time_policy(context)
    scores: dict[str, VelocityErrorScore] = {}
    invalid: dict[str, str] = {}
    prediction_resampled_ids: set[str] = set()
    for sample_id in tqdm(_matched_motion_ids(context, "E_vel"), desc="Computing E_vel", unit="motion", dynamic_ncols=True):
        try:
            prediction_qpos, groundtruth_qpos, hold_frames, prediction_resampled = _physical_metric_qpos_pair(
                context, sample_id
            )
            if prediction_resampled:
                prediction_resampled_ids.add(sample_id)
            scores[sample_id] = evaluator.score(
                prediction_qpos,
                groundtruth_qpos,
                prediction_fps=prediction_fps,
                groundtruth_fps=groundtruth_fps,
                target_fps=physical_fps,
                groundtruth_hold_frames=hold_frames,
            )
        except Exception as exc:
            invalid[sample_id] = str(exc)
    if not scores:
        raise RuntimeError("no valid prediction/GT pairs for E_vel")
    context.invalid["e_vel"] = invalid
    valid_ids = sorted(scores)
    values = np.asarray([scores[key].value for key in valid_ids], dtype=np.float64)
    frames = np.asarray([scores[key].num_frames for key in valid_ids], dtype=np.int32)
    velocity_frames = np.asarray([scores[key].num_velocity_frames for key in valid_ids], dtype=np.int32)
    durations = np.asarray([scores[key].duration_seconds for key in valid_ids], dtype=np.float32)
    holds = np.asarray([scores[key].groundtruth_hold_frames for key in valid_ids], dtype=np.int32)
    np.savez_compressed(
        context.output / "cache" / "e_vel.npz",
        sample_ids=np.asarray(valid_ids, dtype=np.str_),
        values_mm_per_frame=values.astype(np.float32),
        num_frames=frames,
        num_velocity_frames=velocity_frames,
        duration_seconds=durations,
        fps=np.full(len(valid_ids), physical_fps, dtype=np.float32),
        groundtruth_hold_frames=holds,
        prediction_resampled_to_groundtruth_frames=np.asarray(
            [key in prediction_resampled_ids for key in valid_ids], dtype=np.bool_
        ),
        time_policy=np.asarray(time_policy),
    )
    return MetricOutput(float(values.mean()), len(valid_ids), {
        "unit": "mm/frame",
        "definition": "OMG Video/SONIC E_vel: fixed-14-link mean L2 error between prediction and GT first-order body-link displacements",
        "implementation": "omg.benchmarks.metrics.tracking.e_vel",
        "aggregation": "macro_mean_over_complete_matched_motions",
        "time_alignment": time_policy,
        "prediction_time_source": "observed CSV frames at fixed 50 FPS; resampled only within this physical metric when frame counts differ",
        "prediction_resampling": "endpoint-preserving linear translation/joints plus shortest-path quaternion SLERP to effective GT frame count, physical metrics only",
        "num_prediction_resampled_to_groundtruth_frames": len(prediction_resampled_ids),
        "groundtruth_time_source": groundtruth_time_source,
        "groundtruth_hold_source": groundtruth_hold_source,
        "prediction_fps": prediction_fps,
        "motion_groundtruth_fps": groundtruth_fps,
        "physical_fps": physical_fps,
        "position_set": "OMG Video/SONIC fixed 14 G1 FK body-link origins, including pelvis root",
        "body_indices": OMG_SONIC_BODY_INDICES.tolist(),
        "num_joints": int(next(iter(scores.values())).num_joints),
        "mean_evaluated_frames": float(frames.mean()),
        "mean_velocity_frames": float(velocity_frames.mean()),
        "mean_groundtruth_hold_frames": float(holds.mean()),
        "no_root_alignment": True,
        "no_temporal_resampling": False,
        "no_common_prefix_truncation": True,
        "unequal_pairs": "prediction_resampled_to_effective_groundtruth_frame_count_before_scoring",
    }, valid_ids, values)


def _mm_distance(context: EvaluationContext) -> MetricOutput:
    sample_ids, motion, condition, protocol, _ = encode_mm_pairs(context)
    value, distances = paired_mm_distance(condition, motion)
    return MetricOutput(
        value,
        len(sample_ids),
        {
            "unit": "complete_source_pair",
            "aggregation": "mean_paired_euclidean_distance",
            "cross_modal_evaluator": str(context.config["mm_encoder"]),
            "embedding_dim": int(motion.shape[1]),
            **protocol,
        },
        sample_ids,
        distances,
    )


def _cached_metric(
    context: EvaluationContext, name: str, runner: Callable[[EvaluationContext], MetricOutput]
) -> MetricOutput:
    result = context.metric_outputs.get(name)
    if result is None:
        result = runner(context)
        context.metric_outputs[name] = result
    return result


def _bg(context: EvaluationContext) -> MetricOutput:
    fid = _cached_metric(context, "FID", _fid)
    mm_distance = _cached_metric(context, "MM-Distance", _mm_distance)
    value = compute_behavior_generation_score(fid.value, mm_distance.value)
    return MetricOutput(value, min(fid.num_samples, mm_distance.num_samples), {
        "formula": "1000 * exp(-0.3 * FID - 1.6 * MM-Distance)",
        "fid": fid.value, "mm_distance": mm_distance.value,
        "scale": 1000.0, "alpha": 0.3, "beta": 1.6,
        "source_metrics": {"fid": "FID", "mm_distance": "MM-Distance"},
    })


def _bs_level1(context: EvaluationContext) -> MetricOutput:
    bg = _cached_metric(context, "BG", _bg)
    return MetricOutput(bg.value, bg.num_samples, {
        "formula": "BS_level1 = BG * IR_1",
        "bg": bg.value, "ir_1": 1.0, "ir_1_definition": "Level-2 constant",
        "source_metrics": {"bg": "BG"},
    })


def _bs_level2(context: EvaluationContext) -> MetricOutput:
    if context.motion_groundtruth_index is None:
        raise ValueError("BS_level2 requires --motion-groundtruth")
    if not context.level2_task_metadata_by_id:
        raise ValueError("BS_level2 requires a Steerable Motion Benchmark Level-2 task JSON directory")
    families = {"".join(character for character in str(metadata.get("task_family", "")).casefold() if character.isalnum()) for metadata in context.level2_task_metadata_by_id.values()}
    if len(families) != 1:
        raise ValueError(f"BS_level2 requires one Level-2 task family, found: {sorted(families)}")
    family = next(iter(families))
    if family == "speed":
        ir_2 = speed_ir_2(
            context.prediction_index, context.motion_groundtruth_index, context.level2_task_metadata_by_id,
            prediction_fps=float(context.config["prediction_fps"]),
            groundtruth_fps=float(context.config["motion_groundtruth_fps"]),
        )
    elif family == "amplitude":
        ir_2 = amplitude_ir_2(
            context.prediction_index, context.motion_groundtruth_index, context.level2_task_metadata_by_id,
        )
    elif family == "bodyrestrain":
        ir_2 = body_restrain_ir_2(
            context.prediction_index, context.motion_groundtruth_index, context.level2_task_metadata_by_id,
            device=context.device,
        )
    elif family == "direction":
        ir_2 = direction_ir_2(
            context.prediction_index, context.motion_groundtruth_index, context.level2_task_metadata_by_id,
        )
    elif family == "trajectory":
        ir_2 = trajectory_ir_2(
            context.prediction_index, context.motion_groundtruth_index, context.level2_task_metadata_by_id,
        )
    else:
        raise ValueError(
            "IR_2 is currently implemented for Level-2 Speed, Amplitude, BodyRestrain, Direction, and Trajectory only; "
            f"found task family: {family!r}"
        )
    bs_level1 = _cached_metric(context, "BS_level1", _bs_level1)
    values = ir_2.sample_values * bs_level1.value
    return MetricOutput(bs_level1.value * ir_2.value, len(ir_2.sample_ids), {
        "formula": "BS_level2 = BS_level1 * IR_2 = BG * IR_1 * IR_2",
        "bg": bs_level1.value, "bs_level1": bs_level1.value, "ir_1": 1.0, "ir_2": ir_2.value,
        **ir_2.details, "source_metrics": {"bs_level1": "BS_level1"},
    }, ir_2.sample_ids, values)


def _r_at_k(context: EvaluationContext, top_k: int) -> MetricOutput:
    """Formal fixed-candidate Motion-to-Condition Recall@K."""
    import json

    cache = getattr(context, "_retrieval_cache", None)
    if cache is None:
        sample_ids, motion, condition, protocol, grouping = encode_mm_pairs(context)
        batch_size = int(context.config.get("retrieval_batch_size", 32))
        seed = int(context.config.get("retrieval_seed", 0))
        retrieval = fixed_batch_retrieval(
            condition, motion, batch_size=batch_size,
            dataset_names=list(grouping["dataset_names"]), seed=seed,
        )
        used_ids = [sample_ids[int(index)] for index in retrieval.order]
        excluded_ids = [sample_ids[int(index)] for index in retrieval.excluded_indices]
        groups = [
            {
                "group_index": group_index,
                "sample_ids": used_ids[start : start + retrieval.batch_size],
                "dataset_names": [
                    grouping["dataset_names"][int(index)]
                    for index in retrieval.order[start : start + retrieval.batch_size]
                ],
            }
            for group_index, start in enumerate(
                range(0, retrieval.num_used_pairs, retrieval.batch_size)
            )
        ]
        audit_path = context.output / "metrics" / "r_at_groups.json"
        audit_path.write_text(json.dumps({
            "protocol": "OMG_fixed_candidate_groups_non_strict_motion_to_condition",
            "seed": seed,
            "batch_size": retrieval.batch_size,
            "reported_top_k": [1, 5, 10],
            "order_policy": retrieval.order_policy,
            "remainder_policy": "stratified_order_then_drop_incomplete_final_group",
            "excluded_sample_ids": excluded_ids,
            "groups": groups,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cache = (motion, protocol, retrieval, used_ids, excluded_ids, audit_path, seed, grouping)
        setattr(context, "_retrieval_cache", cache)
    motion, protocol, retrieval, used_ids, excluded_ids, audit_path, seed, grouping = cache
    hits = retrieval.hits_at(top_k)
    return MetricOutput(
        retrieval.recall_at(top_k),
        retrieval.num_used_pairs,
        {
            "unit": "retrieval_hit_rate",
            "definition": (
                f"correct paired condition ranks within the nearest {top_k} among "
                "OMG non-strict fixed-size candidate groups"
            ),
            "aggregation": f"mean_top_{top_k}_hit_over_complete_fixed_candidate_groups",
            "cross_modal_evaluator": str(context.config["mm_encoder"]),
            "embedding_dim": int(motion.shape[1]),
            "candidate_batch_size": retrieval.batch_size,
            "top_k": top_k,
            "random_baseline": top_k / retrieval.batch_size,
            "num_matched_pairs": retrieval.num_input_pairs,
            "num_complete_batches": retrieval.num_used_pairs // retrieval.batch_size,
            "num_excluded_remainder": len(excluded_ids),
            "excluded_remainder_policy": "OMG_non_strict_stratified_order_then_drop_incomplete_final_group",
            "retrieval_order_policy": retrieval.order_policy,
            "retrieval_seed": seed,
            "retrieval_group_audit": str(audit_path),
            "query": "generated_motion",
            "candidates": "paired_instruction_conditions",
            **protocol,
        },
        used_ids,
        hits,
    )


def _r_at_1(context: EvaluationContext) -> MetricOutput:
    return _r_at_k(context, 1)


def _r_at_5(context: EvaluationContext) -> MetricOutput:
    return _r_at_k(context, 5)


def _r_at_10(context: EvaluationContext) -> MetricOutput:
    return _r_at_k(context, 10)


def _bas_details(context: EvaluationContext, sample_ids: list[str]) -> dict[str, Any]:
    rows = context.bas_alignment_rows()
    common_frames = np.asarray([rows[key]["common_frames"] for key in sample_ids], dtype=np.int32)
    return {
        "unit": "complete_source_motion",
        "aggregation": "macro_mean_over_matched_source_motions",
        "num_source_samples": len(sample_ids),
        "per_sample_report": str(context.output / "metrics" / "bas_samples.jsonl"),
        "mean_common_frames": float(common_frames.mean()),
        "min_common_frames": int(common_frames.min()),
        "max_common_frames": int(common_frames.max()),
        "audio_beat": "audio29[:, -1] > 0.5",
        "motion_beat": "local_minimum_of_mean_multi_body_displacement",
        **context.bas_protocol(),
    }


def _bas_gen(context: EvaluationContext) -> MetricOutput:
    generated, reference = context.bas_scores()
    sample_ids = sorted(set(generated) & set(reference))
    values = np.asarray([generated[key].value for key in sample_ids], dtype=np.float64)
    gt_values = np.asarray([reference[key].value for key in sample_ids], dtype=np.float64)
    return MetricOutput(float(values.mean()), len(sample_ids), {
        "definition": "BAS-Gen_i = BeatAlign(audio_i, generated_motion_i)",
        "bas_gen_mean": float(values.mean()),
        "bas_gen_std": float(values.std()),
        "bas_gt_mean": float(gt_values.mean()),
        "bas_gt_std": float(gt_values.std()),
        "num_audio_beats": int(sum(generated[key].num_audio_beats for key in sample_ids)),
        "num_generated_motion_beats": int(sum(generated[key].num_motion_beats for key in sample_ids)),
        "num_groundtruth_motion_beats": int(sum(reference[key].num_motion_beats for key in sample_ids)),
        **_bas_details(context, sample_ids),
    }, sample_ids, values)


def _bas_gap(context: EvaluationContext) -> MetricOutput:
    generated, reference = context.bas_scores()
    sample_ids = sorted(set(generated) & set(reference))
    values = np.asarray([bas_gap(generated[key], reference[key]) for key in sample_ids], dtype=np.float64)
    generated_values = np.asarray([generated[key].value for key in sample_ids], dtype=np.float64)
    reference_values = np.asarray([reference[key].value for key in sample_ids], dtype=np.float64)
    return MetricOutput(float(values.mean()), len(sample_ids), {
        "definition": "BAS-Gap_i = BAS-Gen_i - BAS-GT_i",
        "bas_gap_mean": float(values.mean()),
        "bas_gap_std": float(values.std()),
        "generated_mean": float(generated_values.mean()),
        "generated_std": float(generated_values.std()),
        "reference_mean": float(reference_values.mean()),
        "reference_std": float(reference_values.std()),
        **_bas_details(context, sample_ids),
    }, sample_ids, values)


METRICS: dict[str, MetricSpec] = {
    "FID": MetricSpec("FID", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH)), "lower_is_better", _fid),
    "Diversity": MetricSpec("Diversity", frozenset((PREDICTION_MOTION,)), "higher_is_better", _diversity),
    "ContactSliding": MetricSpec("ContactSliding", frozenset((PREDICTION_MOTION,)), "lower_is_better", _contact_sliding),
    "MPJPE": MetricSpec("MPJPE", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH)), "lower_is_better", _mpjpe),
    "g-MPJPE": MetricSpec("g-MPJPE", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH)), "lower_is_better", _g_mpjpe),
    "E_vel": MetricSpec("E_vel", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH)), "lower_is_better", _e_vel),
    "MM-Distance": MetricSpec("MM-Distance", frozenset((PREDICTION_MOTION, INSTRUCTION_GROUNDTRUTH)), "lower_is_better", _mm_distance),
    "BG": MetricSpec("BG", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _bg),
    "BS_level1": MetricSpec("BS_level1", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _bs_level1),
    "BS_level2": MetricSpec("BS_level2", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _bs_level2),
    "R@1": MetricSpec("R@1", frozenset((PREDICTION_MOTION, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _r_at_1),
    "R@5": MetricSpec("R@5", frozenset((PREDICTION_MOTION, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _r_at_5),
    "R@10": MetricSpec("R@10", frozenset((PREDICTION_MOTION, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _r_at_10),
    "BAS-Gen": MetricSpec("BAS-Gen", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH, INSTRUCTION_GROUNDTRUTH)), "higher_is_better", _bas_gen),
    "BAS-Gap": MetricSpec("BAS-Gap", frozenset((PREDICTION_MOTION, MOTION_GROUNDTRUTH, INSTRUCTION_GROUNDTRUTH)), "closer_to_zero", _bas_gap),
}


def resolve_metrics(names: list[str]) -> list[MetricSpec]:
    aliases = {_normal(name): spec for name, spec in METRICS.items()}
    selected: list[MetricSpec] = []
    seen: set[str] = set()
    for name in names:
        key = _normal(name)
        if key not in aliases:
            raise ValueError(f"unknown metric {name!r}; available metrics: {', '.join(METRICS)}")
        spec = aliases[key]
        if spec.name not in seen:
            selected.append(spec)
            seen.add(spec.name)
    if not selected:
        raise ValueError("at least one metric must be selected")
    return selected


def _normal(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum()).replace("level", "")
