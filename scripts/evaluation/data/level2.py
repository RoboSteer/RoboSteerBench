"""Adapters for Steerable Motion Benchmark Level-2 task JSON directories.

Level-2 task roots contain task specifications rather than directly usable
motion CSVs or instruction manifests. This module resolves the original
motion path, modality-specific condition, and task duration for each prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from typing import Any

from .instruction import InstructionIndex, InstructionSample
from .motion import MotionIndex
from .timing import TaskTimingIndex


@dataclass(frozen=True)
class Level2TaskBundle:
    """Level-2 data keyed by the *actual prediction* sample IDs."""

    motion_groundtruth: MotionIndex
    instruction_groundtruth: InstructionIndex
    timing: TaskTimingIndex
    task_metadata_by_id: dict[str, dict[str, Any]]


def load_level2_task_bundle(
    task_root: Path,
    prediction_index: MotionIndex,
    *,
    derived_motion_root: Path | None = None,
    mm_encoder: str | None = None,
    direction_task_type: str | None = None,
) -> Level2TaskBundle | None:
    """Load a Level-2 JSON root, or return ``None`` for ordinary GT roots.

    A prediction exporter may spell the task prefix differently from the JSON
    (for example ``L2_SPEED_TXT`` versus ``L2_speed_text``). The stable key is
    ``basename(ground_truth.motion_parameters) + metadata.task_type``; this is
    also checked for ambiguity instead of silently selecting an arbitrary task.

    ``ground_truth.motion_parameters`` is the authoritative reference-motion
    field. ``metadata.original_motion`` records task provenance and is not
    used to select or load evaluation ground truth.
    """
    task_root = task_root.resolve()
    if not task_root.is_dir():
        return None
    json_paths = sorted(task_root.glob("*.json"))
    if not json_paths:
        return None

    records: list[tuple[Path, dict[str, Any]]] = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid task JSON: {path}") from exc
        if not _is_level2_task(payload):
            continue
        records.append((path, payload))
    if not records:
        return None

    dataset_root = _dataset_root(task_root)
    selected_direction_task_type = _direction_task_type(direction_task_type)
    selected_mm_encoder = _select_mm_encoder(mm_encoder, task_root, records)
    condition_metadata = _condition_index_metadata(selected_mm_encoder)
    by_alias: dict[str, tuple[Path, dict[str, Any]]] = {}
    by_source_id: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, payload in records:
        metadata = payload["metadata"]
        ground_truth = payload["ground_truth"]
        task_id = str(metadata["task_id"])
        source_id = Path(str(ground_truth["motion_parameters"])).name
        task_type = str(metadata["task_type"])
        by_source_id.setdefault(_normal(source_id), []).append((path, payload))
        for alias in (_normal(task_id), _normal(f"{source_id}_{task_type}")):
            existing = by_alias.get(alias)
            if existing is not None and existing[0] != path:
                raise ValueError(
                    f"ambiguous Level-2 task alias {alias!r}: {existing[0]} and {path}"
                )
            by_alias[alias] = (path, payload)
    # Some exporters omit the BodyRestrain arms/legs suffix. The GT motion
    # parameter ID is sufficient only if one JSON task uses that source.
    for source_id, source_records in by_source_id.items():
        if len(source_records) == 1:
            by_alias[source_id] = source_records[0]

    motion_samples: dict[str, Path] = {}
    instruction_samples: dict[str, InstructionSample] = {}
    durations: dict[str, float] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for prediction_id in sorted(prediction_index.samples):
        record = by_alias.get(_normal(prediction_id))
        if record is None:
            # Prediction task prefixes are implementation-specific.  Their
            # suffix remains the original motion ID plus the requested type.
            matches = [
                value for alias, value in by_alias.items()
                if _normal(prediction_id).endswith(alias)
            ]
            unique = {path: payload for path, payload in matches}
            if len(unique) == 1:
                record = next(iter(unique.items()))
            elif len(unique) > 1:
                raise ValueError(f"ambiguous Level-2 task match for prediction {prediction_id!r}")
        if record is None and selected_direction_task_type is not None:
            # A bare source ID is ambiguous for Direction because each source
            # has both a left and a right task. Resolve it only with an
            # explicit side selected by the automatic dual evaluation.
            matches = [
                value
                for value in by_source_id.get(_normal(prediction_id), [])
                if _normalized_name(str(value[1]["metadata"].get("task_family", ""))) == "direction"
                and str(value[1]["metadata"].get("task_type", "")).casefold()
                == selected_direction_task_type
            ]
            if len(matches) == 1:
                record = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    f"ambiguous Level-2 Direction match for prediction {prediction_id!r} "
                    f"and task type {selected_direction_task_type!r}"
                )
        if record is None:
            missing.append(prediction_id)
            continue
        json_path, payload = record
        metadata = payload["metadata"]
        ground_truth = payload["ground_truth"]
        motion_parameters = dataset_root / str(ground_truth["motion_parameters"])
        if not motion_parameters.is_dir():
            raise FileNotFoundError(
                f"{json_path}: ground_truth.motion_parameters resolves to unsupported or missing "
                f"motion directory {motion_parameters}"
            )
        evaluation_motion = motion_parameters
        task_family = _normalized_name(str(metadata.get("task_family", "")))
        if task_family in {"bodyrestrain", "direction"}:
            if derived_motion_root is None:
                raise ValueError(
                    f"Level-2 {metadata['task_family']} requires derived_motion_root for its pseudo GT"
                )
            if task_family == "bodyrestrain":
                evaluation_motion = _materialize_body_restrain_pseudo_gt(
                    source=motion_parameters,
                    task_id=str(metadata["task_id"]),
                    task_type=str(metadata["task_type"]),
                    cache_root=derived_motion_root,
                )
            else:
                evaluation_motion = _materialize_direction_pseudo_gt(
                    source=motion_parameters,
                    task_id=str(metadata["task_id"]),
                    task_type=str(metadata["task_type"]),
                    cache_root=derived_motion_root,
                )
        condition_path, condition_attributes = _level2_condition(
            payload, json_path, dataset_root, selected_mm_encoder
        )
        duration = _duration(metadata.get("duration"), json_path)
        motion_samples[prediction_id] = evaluation_motion
        instruction_samples[prediction_id] = InstructionSample(
            sample_id=prediction_id,
            path=condition_path,
            attributes={
                **condition_attributes,
                "semantic_id": motion_parameters.name,
                "dataset": f"level2_{metadata.get('task_family', 'unknown')}",
                "task_id": str(metadata["task_id"]),
                "task_type": str(metadata["task_type"]),
            },
        )
        durations[prediction_id] = duration
        metadata_by_id[prediction_id] = {
            "task_id": str(metadata["task_id"]),
            "task_family": str(metadata["task_family"]),
            "task_type": str(metadata["task_type"]),
            "task_json": str(json_path),
            "motion_parameters": str(motion_parameters),
            "evaluation_motion": str(evaluation_motion),
            "groundtruth_transform": _groundtruth_transform_name(task_family, evaluation_motion, motion_parameters),
        }
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"{task_root}: no same-task JSON for prediction directories: {preview}{suffix}"
        )
    digest = hashlib.sha256(
        "".join(f"{key}\t{durations[key]:.9f}\n" for key in sorted(durations)).encode("utf-8")
    ).hexdigest()
    return Level2TaskBundle(
        motion_groundtruth=MotionIndex(task_root, motion_samples, {}),
        instruction_groundtruth=InstructionIndex(
            root=task_root,
            encoder=selected_mm_encoder,
            samples=instruction_samples,
            metadata={
                "kind": "level2_task_json",
                **condition_metadata,
                "source": str(task_root),
                "mm_encoder": selected_mm_encoder,
            },
        ),
        timing=TaskTimingIndex(
            source=task_root,
            durations_seconds=durations,
            record_count=len(durations),
            duration_manifest_sha256=digest,
            source_format="level2_task_json_directory",
        ),
        task_metadata_by_id=metadata_by_id,
    )


_LEVEL2_MODALITY_BY_ENCODER = {
    "text_motion": ("text", "text"),
    "audio_motion": ("audio", "audios"),
    "rhythm_motion": ("audio", "audios"),
    "video_motion_human": ("video", "videos_processed"),
    "video_motion_skel": ("video", "videos_processed"),
}


def _direction_task_type(value: str | None) -> str | None:
    """Validate the internal side used for bare Direction prediction IDs."""
    if value is None:
        return None
    selected = value.strip().casefold()
    if selected not in {"left", "right"}:  # defensive API validation
        raise ValueError(f"direction_task_type must be 'left' or 'right', got {value!r}")
    return selected


def _select_mm_encoder(
    requested: str | None,
    task_root: Path,
    records: list[tuple[Path, dict[str, Any]]],
) -> str:
    if requested:
        selected = requested.strip()
        if selected not in _LEVEL2_MODALITY_BY_ENCODER:
            supported = ", ".join(sorted(_LEVEL2_MODALITY_BY_ENCODER))
            raise ValueError(
                f"Level-2 task JSON does not support --mm-encoder {selected!r}; "
                f"expected one of: {supported}"
            )
        return selected

    available: set[str] = set()
    video_values: list[str] = []
    for _, payload in records:
        modalities = payload["input"]["modalities"]
        text = modalities.get("text")
        if isinstance(text, str) and text.strip():
            available.add("text_motion")
        if _nonempty_condition_values(modalities.get("audios")):
            available.add("audio_motion")
        videos = _nonempty_condition_values(modalities.get("videos_processed"))
        if videos:
            video_values.extend(videos)

    if video_values:
        video_hint = " ".join([task_root.name, *video_values]).casefold()
        if "skeleton" in video_hint or "skel" in video_hint:
            available.add("video_motion_skel")
        elif "human" in video_hint:
            available.add("video_motion_human")
        else:
            raise ValueError(
                f"{task_root}: cannot infer whether videos_processed contains human or "
                "skeleton video; pass --mm-encoder explicitly"
            )
    if len(available) == 1:
        return next(iter(available))
    if not available:
        raise ValueError(
            f"{task_root}: Level-2 task JSON has no non-empty text, audios, or "
            "videos_processed condition"
        )
    raise ValueError(
        f"{task_root}: multiple Level-2 condition modalities are populated "
        f"({', '.join(sorted(available))}); pass --mm-encoder explicitly"
    )


def _level2_condition(
    payload: dict[str, Any],
    json_path: Path,
    dataset_root: Path,
    mm_encoder: str,
) -> tuple[Path, dict[str, Any]]:
    modality, field = _LEVEL2_MODALITY_BY_ENCODER[mm_encoder]
    modalities = payload["input"]["modalities"]
    raw_value = modalities.get(field)
    if modality == "text":
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(
                f"{json_path}: --mm-encoder {mm_encoder!r} requires a non-empty "
                f"input.modalities.{field} value"
            )
        return json_path, {
            "text": raw_value.strip(),
            "condition_modality": modality,
            "condition_field": field,
        }

    values = _nonempty_condition_values(raw_value)
    if not values:
        raise ValueError(
            f"{json_path}: --mm-encoder {mm_encoder!r} requires at least one path in "
            f"input.modalities.{field}"
        )
    paths = [_resolve_level2_asset(value, dataset_root, json_path, field) for value in values]
    if modality == "audio":
        if len(paths) != 1:
            raise ValueError(
                f"{json_path}: audio evaluation requires exactly one path in "
                f"input.modalities.{field}, got {len(paths)}"
            )
        return paths[0], {
            "assets": {"raw_audio": str(paths[0])},
            "condition_modality": modality,
            "condition_field": field,
        }

    video_asset: str | list[str]
    if len(paths) == 1:
        video_asset = str(paths[0])
    else:
        video_asset = [str(path) for path in paths]
    return paths[0], {
        "assets": {"video": video_asset},
        "condition_modality": modality,
        "condition_field": field,
    }


def _nonempty_condition_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _resolve_level2_asset(
    value: str, dataset_root: Path, json_path: Path, field: str
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{json_path}: input.modalities.{field} asset does not exist: {path}"
        )
    return path


def _condition_index_metadata(mm_encoder: str) -> dict[str, str]:
    modality, field = _LEVEL2_MODALITY_BY_ENCODER[mm_encoder]
    formats = {
        "text": "inline_text_v1",
        "audio": "level2_audio_asset_v1",
        "video": "level2_video_asset_v1",
    }
    return {
        "format": formats[modality],
        "condition_modality": modality,
        "condition_field": field,
    }


def _is_level2_task(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    ground_truth = payload.get("ground_truth")
    modalities = (payload.get("input") or {}).get("modalities") if isinstance(payload.get("input"), dict) else None
    return (
        isinstance(metadata, dict)
        and str(metadata.get("task_level", "")).casefold() == "level2"
        and all(isinstance(metadata.get(key), str) and metadata[key] for key in ("task_id", "task_type"))
        and isinstance(ground_truth, dict)
        and isinstance(ground_truth.get("motion_parameters"), str)
        and bool(ground_truth["motion_parameters"])
        and isinstance(modalities, dict)
        and any(key in modalities for key in ("text", "audios", "videos_processed"))
    )


BODY_RESTRAIN_DOF_INDICES = {
    "legs": tuple(range(0, 12)),
    "arms": tuple(range(15, 29)),
}
BODY_RESTRAIN_PSEUDO_GT_SCHEMA = "body_restrain_first_frame_hold_v1"


def _materialize_body_restrain_pseudo_gt(
    *, source: Path, task_id: str, task_type: str, cache_root: Path
) -> Path:
    """Create a cache-local pseudo GT with the requested G1 DOFs held at frame zero."""
    normalized_type = task_type.casefold()
    if normalized_type not in BODY_RESTRAIN_DOF_INDICES:
        raise ValueError(
            f"BodyRestrain task_type must be 'arms' or 'legs', got {task_type!r}"
        )
    source = source.resolve()
    joint_path = source / "joint_pos.csv"
    required = (joint_path, source / "body_pos.csv", source / "body_quat.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"BodyRestrain source motion is missing required CSVs: {', '.join(missing)}"
        )
    stat = joint_path.stat()
    fingerprint = hashlib.sha256(
        "\0".join((
            BODY_RESTRAIN_PSEUDO_GT_SCHEMA,
            str(source),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            task_id,
            normalized_type,
            ",".join(map(str, BODY_RESTRAIN_DOF_INDICES[normalized_type])),
        )).encode("utf-8")
    ).hexdigest()[:20]
    cache_root = cache_root.resolve()
    target = cache_root / f"body_restrain_{fingerprint}"
    required_output = tuple(target / name for name in ("joint_pos.csv", "body_pos.csv", "body_quat.csv", "metadata.json"))
    if target.is_dir() and all(path.exists() for path in required_output):
        return target
    if target.exists():
        raise RuntimeError(f"incomplete BodyRestrain pseudo-GT cache entry: {target}")

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".body_restrain_", dir=cache_root))
    try:
        header = joint_path.read_text(encoding="utf-8").splitlines()[0]
        joint = np.loadtxt(joint_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        if joint.ndim != 2 or joint.shape[1] != 29 or joint.shape[0] < 2:
            raise ValueError(f"{joint_path}: expected at least two frames of 29 joint angles")
        if not np.isfinite(joint).all():
            raise ValueError(f"{joint_path}: non-finite joint angle")
        restricted = BODY_RESTRAIN_DOF_INDICES[normalized_type]
        joint[:, restricted] = joint[0, restricted]
        np.savetxt(
            temporary / "joint_pos.csv", joint, delimiter=",", fmt="%.9g",
            header=header, comments="",
        )
        for filename in ("body_pos.csv", "body_quat.csv"):
            (temporary / filename).symlink_to(source / filename)
        (temporary / "metadata.json").write_text(
            json.dumps({
                "schema": BODY_RESTRAIN_PSEUDO_GT_SCHEMA,
                "task_id": task_id,
                "task_type": normalized_type,
                "source_motion_parameters": str(source),
                "restricted_dof_indices": list(restricted),
                "rule": "joint_pos[t, restricted_dof] = joint_pos[0, restricted_dof]",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _groundtruth_transform_name(
    task_family: str, evaluation_motion: Path, motion_parameters: Path
) -> str | None:
    if evaluation_motion == motion_parameters:
        return None
    if task_family == "bodyrestrain":
        return "body_restrain_restricted_dofs_held_at_first_frame"
    if task_family == "direction":
        return "direction_root_xy_rotated_to_target_side"
    raise ValueError(f"unexpected pseudo-GT task family {task_family!r}")


def _materialize_direction_pseudo_gt(
    *, source: Path, task_id: str, task_type: str, cache_root: Path
) -> Path:
    """Create a cache-local pseudo GT whose root XY endpoint targets local left/right."""
    target_side = task_type.casefold()
    if target_side not in {"left", "right"}:
        raise ValueError(f"Direction task_type must be 'left' or 'right', got {task_type!r}")
    source = source.resolve()
    body_pos_path = source / "body_pos.csv"
    body_quat_path = source / "body_quat.csv"
    joint_path = source / "joint_pos.csv"
    required = (body_pos_path, body_quat_path, joint_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Direction source motion is missing required CSVs: {', '.join(missing)}"
        )
    fingerprint = hashlib.sha256(
        "\0".join((
            "direction_root_xy_rotate_v1",
            str(source),
            str(body_pos_path.stat().st_size),
            str(body_pos_path.stat().st_mtime_ns),
            str(body_quat_path.stat().st_size),
            str(body_quat_path.stat().st_mtime_ns),
            task_id,
            target_side,
        )).encode("utf-8")
    ).hexdigest()[:20]
    cache_root = cache_root.resolve()
    target = cache_root / f"direction_{fingerprint}"
    required_output = tuple(target / name for name in ("joint_pos.csv", "body_pos.csv", "body_quat.csv", "metadata.json"))
    if target.is_dir() and all(path.exists() for path in required_output):
        return target
    if target.exists():
        raise RuntimeError(f"incomplete Direction pseudo-GT cache entry: {target}")

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".direction_", dir=cache_root))
    try:
        header = body_pos_path.read_text(encoding="utf-8").splitlines()[0]
        body_pos = np.loadtxt(body_pos_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        body_quat = np.loadtxt(body_quat_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        joint = np.loadtxt(joint_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        frames = min(len(body_pos), len(body_quat), len(joint))
        if body_pos.ndim != 2 or body_pos.shape[1] < 3 or frames < 2:
            raise ValueError(f"{body_pos_path}: expected at least two frames of root XYZ")
        if body_quat.ndim != 2 or body_quat.shape[1] < 4:
            raise ValueError(f"{body_quat_path}: expected root quaternion wxyz")
        if not np.isfinite(body_pos).all() or not np.isfinite(body_quat).all():
            raise ValueError("Direction source motion contains non-finite root state")
        root_xy = body_pos[:frames, :2]
        displacement = root_xy[-1] - root_xy[0]
        distance = float(np.linalg.norm(displacement))
        if distance < 1e-8:
            raise ValueError(
                f"{source}: cannot derive a Direction pseudo GT from zero final root displacement"
            )
        quaternion = body_quat[0, :4].astype(np.float64)
        quaternion /= np.linalg.norm(quaternion)
        w, x, y, z = quaternion
        forward = np.array((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w)))
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-8:
            raise ValueError(f"{source}: initial root forward axis has no horizontal component")
        forward /= forward_norm
        target_direction = np.array((-forward[1], forward[0]))
        if target_side == "right":
            target_direction *= -1.0
        source_direction = displacement / distance
        angle = float(np.arctan2(
            source_direction[0] * target_direction[1] - source_direction[1] * target_direction[0],
            np.dot(source_direction, target_direction),
        ))
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float32)
        body_pos[:frames, :2] = root_xy[0] + (root_xy - root_xy[0]) @ rotation.T
        np.savetxt(
            temporary / "body_pos.csv", body_pos, delimiter=",", fmt="%.9g",
            header=header, comments="",
        )
        for filename in ("body_quat.csv", "joint_pos.csv"):
            (temporary / filename).symlink_to(source / filename)
        (temporary / "metadata.json").write_text(
            json.dumps({
                "schema": "direction_root_xy_rotate_v1",
                "task_id": task_id,
                "task_type": target_side,
                "source_motion_parameters": str(source),
                "rule": "rotate root_xy[t] - root_xy[0] to target local left/right; retain root quaternion and joint_pos",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _dataset_root(task_root: Path) -> Path:
    for ancestor in (task_root, *task_root.parents):
        if ancestor.name == "Tasks":
            return ancestor.parent
    raise ValueError(
        f"{task_root}: cannot infer dataset root; expected it below "
        "<Steerable Motion Benchmark Dataset>/Tasks/..."
    )


def _duration(value: Any, path: Path) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: metadata.duration must be a positive finite number") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{path}: metadata.duration must be a positive finite number")
    return duration


def _normal(value: str) -> str:
    return value.casefold()


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
