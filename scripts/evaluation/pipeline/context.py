"""Lazy, shared data context for metric execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from scripts.evaluation.core_metric.BAS_Gen import BASGenEvaluator, BASGenScore
from scripts.evaluation.core_metric.ContactSliding import ContactSlidingEvaluator, ContactSlidingScore
from scripts.evaluation.data.instruction import InstructionIndex, load_instruction_array, load_instruction_index
from scripts.evaluation.data.level2 import load_level2_task_bundle
from scripts.evaluation.data.motion import MotionIndex, index_motion_root, load_qpos_36
from scripts.evaluation.data.pairing import build_sample_manifest
from scripts.evaluation.data.timing import TaskTimingIndex, load_task_timing_index
from scripts.evaluation.encoders.loader import RegisteredEncoder, load_encoder
from scripts.evaluation.preprocessing.motion import PreparedMotion, prepare_motion_index, source_sample_id
from scripts.evaluation.preprocessing.phase import prepare_phase_motion_index
from scripts.evaluation.preprocessing.representation import motion_representation
from scripts.evaluation.preprocessing.temporal import resample_qpos
from scripts.evaluation.runtime.motion import ResampledMotionIndex, batch_motion_sequences, load_resampled_motion_index


class EvaluationContext:
    def __init__(self, *, prediction: Path, motion_groundtruth: Path | None,
                 instruction_groundtruth: Path | None, task_metadata: Path, output: Path,
                 models: Path, config: dict[str, Any], device: str,
                 direction_task_type: str | None = None) -> None:
        self.output = output.resolve()
        self.models = models.resolve()
        self.config = config
        self.device = device
        self.task_timing: TaskTimingIndex = load_task_timing_index(task_metadata)
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("cache", "embeddings", "metrics"):
            (self.output / name).mkdir(exist_ok=True)
        self.prediction_index = index_motion_root(prediction)
        self.level2_task_metadata_by_id: dict[str, dict[str, Any]] = {}
        level2_bundle = (
            None if motion_groundtruth is None
            else load_level2_task_bundle(
                motion_groundtruth,
                self.prediction_index,
                derived_motion_root=self.output / "cache" / "level2_pseudo_gt",
                mm_encoder=str(config.get("mm_encoder", "")).strip() or None,
                direction_task_type=direction_task_type,
            )
        )
        if level2_bundle is not None:
            if instruction_groundtruth is not None and instruction_groundtruth.resolve() != motion_groundtruth.resolve():
                raise ValueError(
                    "Level-2 task JSON evaluation requires --motion-groundtruth and "
                    "--instruction-groundtruth to reference the same task JSON directory"
                )
            self.motion_groundtruth_index = level2_bundle.motion_groundtruth
            self.instruction_index = level2_bundle.instruction_groundtruth
            self.task_timing = level2_bundle.timing
            self.level2_task_metadata_by_id = level2_bundle.task_metadata_by_id
        else:
            self.motion_groundtruth_index = (
                None if motion_groundtruth is None else index_motion_root(motion_groundtruth)
            )
            self.instruction_index = (
                None if instruction_groundtruth is None else load_instruction_index(instruction_groundtruth)
            )
        self.manifest_rows = build_sample_manifest(
            self.prediction_index, self.motion_groundtruth_index, self.instruction_index,
            self.output / "manifest.jsonl", task_timing=self.task_timing,
        )
        self._prepared_prediction: PreparedMotion | None = None
        self._prepared_groundtruth: PreparedMotion | None = None
        self._phase_prepared_prediction: PreparedMotion | None = None
        self._phase_prepared_groundtruth: PreparedMotion | None = None
        self._resampled_motion: dict[tuple[str, float, float, tuple[str, ...] | None], ResampledMotionIndex] = {}
        self._motion_encoder: RegisteredEncoder | None = None
        self._prediction_embeddings: dict[str, np.ndarray] | None = None
        self._groundtruth_embeddings: dict[str, np.ndarray] | None = None
        self._phase_prediction_embeddings: dict[str, np.ndarray] | None = None
        self._phase_groundtruth_embeddings: dict[str, np.ndarray] | None = None
        self._instruction_embeddings: dict[str, np.ndarray] | None = None
        self._instruction_audio_features: dict[str, np.ndarray] | None = None
        self._prediction_contact_sliding: dict[str, ContactSlidingScore] | None = None
        self._prediction_bas_gen: dict[str, BASGenScore] | None = None
        self._groundtruth_bas_gen: dict[str, BASGenScore] | None = None
        self._bas_alignment_rows: dict[str, dict[str, Any]] | None = None
        self.invalid: dict[str, dict[str, str]] = {}
        # Metric adapters can reuse previously computed source metrics (BG and
        # BehaviorSteer) without re-running costly FID/MM model inference.
        self.metric_outputs: dict[str, Any] = {}
        self._prediction_metric_sample_ids = set(self.prediction_index.samples)
        if self.motion_groundtruth_index is not None:
            self._prediction_metric_sample_ids &= set(self.motion_groundtruth_index.samples)
            excluded_ids = sorted(
                set(self.prediction_index.samples) - self._prediction_metric_sample_ids
            )
            if excluded_ids:
                self.invalid["prediction_without_motion_groundtruth"] = {
                    sample_id: (
                        "excluded from GT-dependent prediction metrics: "
                        "no matching motion groundtruth"
                    )
                    for sample_id in excluded_ids
                }

    def prediction_motion_embeddings(self) -> dict[str, np.ndarray]:
        if self._prediction_embeddings is None:
            self._prediction_embeddings = self._encode_motion(self._prepare_prediction(), "prediction")
        return self._prediction_embeddings

    def groundtruth_motion_embeddings(self) -> dict[str, np.ndarray]:
        if self.motion_groundtruth_index is None:
            raise ValueError("selected metrics require --motion-groundtruth")
        if self._groundtruth_embeddings is None:
            self._groundtruth_embeddings = self._encode_motion(
                self._prepare_groundtruth(), "motion_groundtruth"
            )
        return self._groundtruth_embeddings

    def prediction_phase_embeddings(self) -> dict[str, np.ndarray]:
        """One phase-normalized embedding per complete generated motion."""
        if self._phase_prediction_embeddings is None:
            self._phase_prediction_embeddings = self._encode_motion(
                self._prepare_phase_prediction(), "phase_prediction"
            )
        return self._phase_prediction_embeddings

    def groundtruth_phase_embeddings(self) -> dict[str, np.ndarray]:
        """One phase-normalized embedding per complete reference motion."""
        if self.motion_groundtruth_index is None:
            raise ValueError("selected metrics require --motion-groundtruth")
        if self._phase_groundtruth_embeddings is None:
            self._phase_groundtruth_embeddings = self._encode_motion(
                self._prepare_phase_groundtruth(), "phase_motion_groundtruth"
            )
        return self._phase_groundtruth_embeddings

    def prediction_contact_sliding(self) -> dict[str, ContactSlidingScore]:
        if self._prediction_contact_sliding is not None:
            return self._prediction_contact_sliding
        protocol = self.contact_sliding_protocol()
        evaluator = ContactSlidingEvaluator(
            device=self.device,
            contact_height_threshold=float(protocol["contact_height_threshold"]),
            contact_penetration_tolerance=float(protocol["contact_penetration_tolerance"]),
        )
        prepared = self._resampled(
            self.prediction_index, float(self.config["prediction_fps"]), "prediction",
            include_sample_ids=set(self._prediction_metric_sample_ids),
            target_fps=float(protocol["target_fps"]),
        )
        scores: dict[str, ContactSlidingScore] = {}
        max_frames = int(self.config.get("max_frames_per_batch", 16384))
        for batch in tqdm(
            batch_motion_sequences(prepared.qpos_by_id, max_frames_per_batch=max_frames),
            desc="Computing ContactSliding", unit="batch", dynamic_ncols=True,
        ):
            batch_scores = evaluator.score_batch(
                batch.qpos_36, valid=batch.valid, fps=float(protocol["target_fps"])
            )
            scores.update(zip(batch.sample_ids, batch_scores, strict=True))
        if not scores:
            raise RuntimeError("no valid prediction motions for ContactSliding")
        self.invalid["prediction_contact_sliding"] = dict(prepared.invalid)
        ids = sorted(scores)
        np.savez_compressed(
            self.output / "cache" / "prediction_contact_sliding.npz",
            sample_ids=np.asarray(ids, dtype=np.str_),
            values=np.asarray([scores[key].value for key in ids], dtype=np.float32),
            num_contact_intervals=np.asarray([scores[key].num_contact_intervals for key in ids], dtype=np.int64),
            num_valid_intervals=np.asarray([scores[key].num_valid_intervals for key in ids], dtype=np.int64),
            fps=np.full(len(ids), float(protocol["target_fps"]), dtype=np.float32),
            task_duration_seconds=np.asarray([self.task_timing.require([key], label="ContactSliding")[key] for key in ids], dtype=np.float32),
            time_reference=np.asarray("prediction_observed_csv_timeline"),
        )
        self._prediction_contact_sliding = scores
        return scores

    def instruction_embeddings(self) -> dict[str, np.ndarray]:
        if self.instruction_index is None:
            raise ValueError("selected metrics require --instruction-groundtruth")
        if self._instruction_embeddings is not None:
            return self._instruction_embeddings
        encoder = load_encoder(self.instruction_index.encoder, self.models, self.device)
        ids: list[str] = []
        values: list[np.ndarray] = []
        invalid: dict[str, str] = {}
        for sample_id, sample in sorted(self.instruction_index.samples.items()):
            try:
                values.append(load_instruction_array(sample))
                ids.append(sample_id)
            except Exception as exc:
                invalid[sample_id] = str(exc)
        if not values:
            raise RuntimeError("no valid instruction groundtruth samples")
        embeddings = encoder.encode(values)
        self._save_embeddings("instruction", ids, embeddings)
        self.invalid["instruction_groundtruth"] = invalid
        self._instruction_embeddings = dict(zip(ids, embeddings, strict=True))
        return self._instruction_embeddings

    def instruction_embeddings_for_prediction_windows(self) -> dict[str, np.ndarray]:
        instruction = self.instruction_embeddings()
        return {
            window_id: instruction[source_sample_id(window_id)]
            for window_id in self.prediction_motion_embeddings()
            if source_sample_id(window_id) in instruction
        }

    def instruction_audio_features(self) -> dict[str, np.ndarray]:
        """Load BAS audio features; the final feature channel stores beats."""
        if self.instruction_index is None:
            raise ValueError("selected metrics require --instruction-groundtruth")
        if self._instruction_audio_features is not None:
            return self._instruction_audio_features
        values: dict[str, np.ndarray] = {}
        invalid: dict[str, str] = {}
        for sample_id, sample in sorted(self.instruction_index.samples.items()):
            try:
                features = load_instruction_array(sample)
                if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
                    raise ValueError(f"audio features must have shape (T, D), got {features.shape}")
                values[sample_id] = features
            except Exception as exc:
                invalid[sample_id] = str(exc)
        if not values:
            raise RuntimeError("no valid audio-feature instruction groundtruth samples")
        self.invalid["instruction_audio_features"] = invalid
        self._instruction_audio_features = values
        return values

    def bas_scores(self) -> tuple[dict[str, BASGenScore], dict[str, BASGenScore]]:
        """Compute BAS-Gen and BAS-GT on OMG's shared raw 50 Hz prefix."""
        if self._prediction_bas_gen is not None and self._groundtruth_bas_gen is not None:
            return self._prediction_bas_gen, self._groundtruth_bas_gen
        if self.motion_groundtruth_index is None:
            raise ValueError("formal OMG BAS evaluation requires --motion-groundtruth")
        protocol = self.bas_protocol()
        if not all(np.isclose(float(protocol[key]), 50.0) for key in (
            "prediction_fps", "motion_groundtruth_fps", "instruction_groundtruth_fps",
        )):
            raise ValueError("OMG BAS formal protocol requires prediction, GT, and audio features at 50 FPS")
        audio = self.instruction_audio_features()
        common_ids = set(self.prediction_index.samples) & set(self.motion_groundtruth_index.samples) & set(audio)
        evaluator = BASGenEvaluator(device=self.device)
        generated: dict[str, BASGenScore] = {}
        reference: dict[str, BASGenScore] = {}
        rows: dict[str, dict[str, Any]] = {}
        invalid: dict[str, str] = {}
        for sample_id in tqdm(sorted(common_ids), desc="Computing OMG BAS", unit="motion", dynamic_ncols=True):
            try:
                prediction_qpos = load_qpos_36(self.prediction_index.samples[sample_id])
                reference_qpos = load_qpos_36(self.motion_groundtruth_index.samples[sample_id])
                features = audio[sample_id]
                common_frames = min(len(features), len(reference_qpos), len(prediction_qpos))
                if common_frames < 2:
                    raise ValueError("prediction, GT, and audio must share at least two 50 Hz frames")
                kwargs = {
                    "motion_fps": 50.0,
                    "audio_fps": 50.0,
                    "beat_threshold": float(protocol["beat_threshold"]),
                    "sigma_frames": float(protocol["sigma_frames"]),
                    "min_motion_beat_distance_seconds": float(protocol["min_motion_beat_distance_seconds"]),
                }
                generated[sample_id] = evaluator.score(prediction_qpos[:common_frames], features[:common_frames], **kwargs)
                reference[sample_id] = evaluator.score(reference_qpos[:common_frames], features[:common_frames], **kwargs)
                rows[sample_id] = {
                    "sample_id": sample_id,
                    "common_frames": common_frames,
                    "audio_feature_frames": int(len(features)),
                    "groundtruth_frames": int(len(reference_qpos)),
                    "prediction_frames": int(len(prediction_qpos)),
                    "bas_gen": generated[sample_id].value,
                    "bas_gt": reference[sample_id].value,
                    "bas_gap": generated[sample_id].value - reference[sample_id].value,
                    "num_audio_beats": generated[sample_id].num_audio_beats,
                    "num_generated_motion_beats": generated[sample_id].num_motion_beats,
                    "num_groundtruth_motion_beats": reference[sample_id].num_motion_beats,
                }
            except Exception as exc:
                invalid[sample_id] = str(exc)
        if not generated:
            raise RuntimeError("no valid matched prediction/GT/audio samples for OMG BAS")
        self.invalid["omg_bas"] = invalid
        self._prediction_bas_gen = generated
        self._groundtruth_bas_gen = reference
        self._bas_alignment_rows = rows
        self._save_bas_artifacts(generated, reference, rows)
        return generated, reference

    def prediction_bas_gen(self) -> dict[str, BASGenScore]:
        return self.bas_scores()[0]

    def groundtruth_bas_gen(self) -> dict[str, BASGenScore]:
        return self.bas_scores()[1]

    def bas_alignment_rows(self) -> dict[str, dict[str, Any]]:
        self.bas_scores()
        assert self._bas_alignment_rows is not None
        return self._bas_alignment_rows

    def aligned(self, left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, label: str
                ) -> tuple[list[str], np.ndarray, np.ndarray]:
        ids = sorted(set(left) & set(right))
        if not ids:
            raise RuntimeError(f"no samples are shared by {label}")
        return ids, np.stack([left[key] for key in ids]), np.stack([right[key] for key in ids])

    def window_protocol(self) -> dict[str, int | float]:
        return {"target_fps": float(self.config["target_fps"]),
                "window_frames": int(self.config["window_frames"]),
                "num_windows": int(self.config["num_windows"])}

    def phase_embedding_protocol(self) -> dict[str, int | str]:
        return {
            "time_policy": "task_duration_phase_normalized",
            "task_duration_source": "metadata.duration",
            "target_frames": int(self.config.get("fid_phase_frames", 60)),
            "motion_samples_per_source": 1,
            "quaternion_interpolation": "endpoint_nlerp_w_positive",
        }

    def contact_sliding_protocol(self) -> dict[str, float]:
        return {
            "target_fps": float(self.config.get("contact_target_fps", 50)),
            "contact_height_threshold": float(self.config.get("contact_height_threshold", 0.12)),
            "contact_penetration_tolerance": float(self.config.get("contact_penetration_tolerance", 0.02)),
        }

    def bas_protocol(self) -> dict[str, float | str | int]:
        return {
            "protocol": "OMG_MUL_RHY_50Hz_v2_single_model",
            "audio_feature_protocol": "OMG.prepare_mul_rhy_omg_artifacts.audio29",
            "audio_feature_n_fft": 1024,
            "time_policy": "raw_50hz_common_prefix_over_audio_gt_and_current_prediction",
            "direction": "music_to_motion",
            "prediction_fps": float(self.config["prediction_fps"]),
            "motion_groundtruth_fps": float(self.config["motion_groundtruth_fps"]),
            "instruction_groundtruth_fps": float(self.config.get("instruction_groundtruth_fps", 50)),
            "beat_threshold": float(self.config.get("bas_beat_threshold", 0.5)),
            "sigma_frames": float(self.config.get("bas_sigma_frames", 3.0)),
            "min_motion_beat_distance_seconds": float(self.config.get("bas_min_motion_beat_distance_seconds", 0.25)),
        }

    def source_count(self, window_ids: list[str]) -> int:
        return len({source_sample_id(item) for item in window_ids})

    def task_duration_seconds(self, sample_id: str, *, label: str) -> float:
        return self.task_timing.require((sample_id,), label=label)[sample_id]

    def report(self) -> dict[str, Any]:
        return {
            "prediction": _index_report(self.prediction_index),
            "motion_groundtruth": _index_report(self.motion_groundtruth_index),
            "instruction_groundtruth": None if self.instruction_index is None else {
                "root": str(self.instruction_index.root), "encoder": self.instruction_index.encoder,
                "num_indexed": len(self.instruction_index.samples),
            },
            "window_protocol": self.window_protocol(),
            "phase_embedding_protocol": self.phase_embedding_protocol(),
            "motion_encoder_batch_size": self._get_motion_encoder().batch_size,
            "contact_sliding_protocol": self.contact_sliding_protocol(),
            "bas_protocol": self.bas_protocol(),
            "task_timing": self.task_timing.report(self._timing_selected_ids()),
            "time_axis_policy": {
                "phase_embedding": "task_metadata.duration phase-normalized to inclusive 60-frame sequence",
                "groundtruth_real_time": "task_metadata.duration",
                "prediction_real_time": "observed CSV frames and declared prediction_fps",
            },
            "invalid": self.invalid,
        }

    def _resampled(
        self, index: MotionIndex, source_fps: float, label: str,
        include_sample_ids: set[str] | None = None, target_fps: float | None = None,
    ) -> ResampledMotionIndex:
        target_fps = float(self.config["target_fps"] if target_fps is None else target_fps)
        included = None if include_sample_ids is None else tuple(sorted(include_sample_ids))
        key = (str(index.root), float(source_fps), target_fps, included)
        if key not in self._resampled_motion:
            self._resampled_motion[key] = load_resampled_motion_index(
                index, source_fps=source_fps, target_fps=target_fps,
                include_sample_ids=include_sample_ids,
                workers=int(self.config.get("preprocess_workers", 0)),
                description=f"Preparing {index.root.name}",
            )
        return self._resampled_motion[key]


    def _prepare_prediction(self) -> PreparedMotion:
        if self._prepared_prediction is None:
            self._prepared_prediction = self._prepare(
                self.prediction_index, float(self.config["prediction_fps"]), "prediction",
                set(self._prediction_metric_sample_ids),
            )
        return self._prepared_prediction

    def _prepare_groundtruth(self) -> PreparedMotion:
        if self.motion_groundtruth_index is None:
            raise ValueError("--motion-groundtruth is required")
        if self._prepared_groundtruth is None:
            self._prepared_groundtruth = self._prepare(
                self.motion_groundtruth_index, float(self.config["motion_groundtruth_fps"]),
                "motion_groundtruth", set(self._prediction_metric_sample_ids),
            )
        return self._prepared_groundtruth

    def _prepare_phase_prediction(self) -> PreparedMotion:
        if self._phase_prepared_prediction is None:
            self._phase_prepared_prediction = self._prepare_phase(
                self.prediction_index, "phase_prediction", set(self._prediction_metric_sample_ids)
            )
        return self._phase_prepared_prediction

    def _prepare_phase_groundtruth(self) -> PreparedMotion:
        if self.motion_groundtruth_index is None:
            raise ValueError("--motion-groundtruth is required")
        if self._phase_prepared_groundtruth is None:
            self._phase_prepared_groundtruth = self._prepare_phase(
                self.motion_groundtruth_index,
                "phase_motion_groundtruth",
                set(self._prediction_metric_sample_ids),
            )
        return self._phase_prepared_groundtruth

    def _prepare(self, index: MotionIndex, source_fps: float, label: str,
                 include_sample_ids: set[str] | None = None) -> PreparedMotion:
        prepared = prepare_motion_index(
            index, source_fps=source_fps, target_fps=float(self.config["target_fps"]),
            window_frames=int(self.config["window_frames"]), num_windows=int(self.config["num_windows"]),
            include_sample_ids=include_sample_ids,
            resampled=self._resampled(index, source_fps, label, include_sample_ids),
        )
        self.invalid[label] = prepared.invalid
        np.savez_compressed(
            self.output / "cache" / f"{label}_qpos.npz",
            sample_ids=np.asarray(prepared.sample_ids, dtype=np.str_),
            source_sample_ids=np.asarray(prepared.source_sample_ids, dtype=np.str_),
            window_indices=prepared.window_indices, window_starts=prepared.window_starts,
            qpos_36=prepared.qpos_36,
            fps=np.full(len(prepared.sample_ids), float(self.config["target_fps"]), np.float32),
        )
        return prepared

    def _prepare_phase(
        self, index: MotionIndex, label: str, include_sample_ids: set[str] | None = None,
    ) -> PreparedMotion:
        selected_ids = set(index.samples)
        if include_sample_ids is not None:
            selected_ids &= include_sample_ids
        durations = self.task_timing.require(selected_ids, label=label)
        prepared = prepare_phase_motion_index(
            index,
            target_frames=int(self.config.get("fid_phase_frames", 60)),
            include_sample_ids=selected_ids,
            workers=int(self.config.get("preprocess_workers", 0)),
            description=f"Phase-normalizing {index.root.name} for distribution metrics",
            duration_seconds_by_id=durations,
        )
        self.invalid[label] = prepared.invalid
        np.savez_compressed(
            self.output / "cache" / f"{label}_qpos.npz",
            sample_ids=np.asarray(prepared.sample_ids, dtype=np.str_),
            source_sample_ids=np.asarray(prepared.source_sample_ids, dtype=np.str_),
            qpos_36=prepared.qpos_36,
            target_frames=np.asarray(int(self.config.get("fid_phase_frames", 60)), dtype=np.int32),
            time_policy=np.asarray("task_duration_phase_normalized"),
            task_duration_seconds=np.asarray([durations[key] for key in prepared.source_sample_ids], dtype=np.float32),
            task_duration_source=np.asarray("metadata.duration"),
        )
        return prepared

    def _get_motion_encoder(self) -> RegisteredEncoder:
        if self._motion_encoder is None:
            self._motion_encoder = load_encoder("motion_encoder", self.models, self.device)
        return self._motion_encoder

    def _encode_motion(self, prepared: PreparedMotion, label: str) -> dict[str, np.ndarray]:
        """Run FK conversion and MotionEncoder inference in the same bounded batch.

        Keeping only one batch's qpos, FK intermediates, local representation,
        and embedding on CUDA prevents large evaluation sets from exhausting GPU
        memory.  The representation cache is intentionally omitted: qpos and
        embeddings are persisted, while local features are reproducible and can
        otherwise occupy multiple gigabytes on the host.
        """
        encoder = self._get_motion_encoder()
        batch_size = encoder.batch_size
        outputs: list[np.ndarray] = []
        total_batches = (len(prepared.qpos_36) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(prepared.qpos_36), batch_size),
            total=total_batches,
            desc=f"Encoding {encoder.name} ({label})",
            unit="batch",
            dynamic_ncols=True,
        ):
            qpos_batch = prepared.qpos_36[start : start + batch_size]
            representation = motion_representation(qpos_batch, encoder.input_key, str(encoder.device))
            outputs.append(encoder.encode_batch(representation))
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        task_durations = None
        if label in {"phase_prediction", "phase_motion_groundtruth"}:
            task_durations = np.asarray(
                [self.task_duration_seconds(key, label=label) for key in prepared.source_sample_ids],
                dtype=np.float32,
            )
        self._save_embeddings(label + "_motion", prepared.sample_ids, embeddings,
                              source_sample_ids=prepared.source_sample_ids,
                              window_indices=prepared.window_indices,
                              window_starts=prepared.window_starts,
                              task_duration_seconds=task_durations)
        return dict(zip(prepared.sample_ids, embeddings, strict=True))

    def _save_bas_artifacts(
        self, generated: dict[str, BASGenScore], reference: dict[str, BASGenScore],
        rows: dict[str, dict[str, Any]],
    ) -> None:
        sample_ids = sorted(generated)
        np.savez_compressed(
            self.output / "cache" / "omg_bas_scores.npz",
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            bas_gen=np.asarray([generated[key].value for key in sample_ids], dtype=np.float32),
            bas_gt=np.asarray([reference[key].value for key in sample_ids], dtype=np.float32),
            bas_gap=np.asarray([generated[key].value - reference[key].value for key in sample_ids], dtype=np.float32),
            common_frames=np.asarray([rows[key]["common_frames"] for key in sample_ids], dtype=np.int32),
            num_audio_beats=np.asarray([generated[key].num_audio_beats for key in sample_ids], dtype=np.int32),
            num_generated_motion_beats=np.asarray([generated[key].num_motion_beats for key in sample_ids], dtype=np.int32),
            num_groundtruth_motion_beats=np.asarray([reference[key].num_motion_beats for key in sample_ids], dtype=np.int32),
        )
        with (self.output / "metrics" / "bas_samples.jsonl").open("w", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                handle.write(json.dumps(rows[sample_id], ensure_ascii=False, sort_keys=True) + "\n")

    def _save_embeddings(self, label: str, sample_ids: list[str] | tuple[str, ...],
                         values: np.ndarray, *, source_sample_ids: tuple[str, ...] | None = None,
                         window_indices: np.ndarray | None = None,
                         window_starts: np.ndarray | None = None,
                         task_duration_seconds: np.ndarray | None = None) -> None:
        payload: dict[str, np.ndarray] = {
            "sample_ids": np.asarray(sample_ids, dtype=np.str_),
            "embeddings": np.asarray(values, dtype=np.float32),
        }
        if source_sample_ids is not None:
            payload["source_sample_ids"] = np.asarray(source_sample_ids, dtype=np.str_)
        if window_indices is not None:
            payload["window_indices"] = np.asarray(window_indices, dtype=np.int32)
        if window_starts is not None:
            payload["window_starts"] = np.asarray(window_starts, dtype=np.int32)
        if task_duration_seconds is not None:
            payload["task_duration_seconds"] = np.asarray(task_duration_seconds, dtype=np.float32)
            payload["task_duration_source"] = np.asarray("metadata.duration")
        np.savez_compressed(self.output / "embeddings" / f"{label}.npz", **payload)


    def _timing_selected_ids(self) -> set[str]:
        return set(self._prediction_metric_sample_ids)

def _index_report(index: MotionIndex | None) -> dict[str, Any] | None:
    if index is None:
        return None
    return {"root": str(index.root), "num_indexed": len(index.samples),
            "duplicates": {key: [str(path) for path in paths] for key, paths in index.duplicates.items()}}
