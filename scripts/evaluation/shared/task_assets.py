"""Resolve benchmark task JSON into explicit, auditable evaluation assets.

The public benchmark stores the authoritative GT and condition paths inside
task JSON.  This module makes those relationships explicit, instead of
inferring them from result-directory names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.evaluation.data.motion import REQUIRED_CSV, normalize_sample_id, sample_id_from_clip_directory


_MODALITY_FIELDS = {
    "text": ("text_motion", "text"),
    "audio": ("audio_motion", "audios"),
    "rhythm": ("rhythm_motion", "audios"),
    "trajectory": ("trajectory_motion", "spatial_coordinates"),
    "human_video": ("video_motion_human", "videos_processed"),
    "skeleton_video": ("video_motion_skel", "videos_processed"),
}


@dataclass(frozen=True)
class ConditionSpec:
    """One resolved original Level-1 condition."""

    modality: str
    encoder: str
    field: str
    text: str | None
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class TaskRecord:
    """One task JSON, keyed by its base/source motion ID."""

    sample_id: str
    task_id: str
    task_level: str
    task_family: str
    task_type: str
    duration_seconds: float
    task_json: Path
    motion_groundtruth: Path
    modalities: dict[str, Any]


def supported_modalities() -> tuple[str, ...]:
    return tuple(_MODALITY_FIELDS)


def index_task_records(task_root: Path, dataset_root: Path) -> list[TaskRecord]:
    """Read every task JSON below ``task_root`` and resolve its GT directory."""
    task_root = task_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not task_root.is_dir():
        raise NotADirectoryError(f"task root does not exist: {task_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root does not exist: {dataset_root}")
    records: list[TaskRecord] = []
    for path in sorted(task_root.rglob("*.json")):
        payload = _read_task_json(path)
        metadata = _require_mapping(payload, "metadata", path)
        ground_truth = _require_mapping(payload, "ground_truth", path)
        task_id = _require_string(metadata, "task_id", path)
        relative_motion = _require_string(ground_truth, "motion_parameters", path)
        motion = _resolve_dataset_path(relative_motion, dataset_root, path)
        if not motion.is_dir():
            raise FileNotFoundError(f"{path}: GT motion directory does not exist: {motion}")
        if not all((motion / filename).is_file() for filename in REQUIRED_CSV):
            raise ValueError(f"{path}: GT motion lacks required CSV files: {motion}")
        try:
            duration = float(metadata["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: missing or invalid metadata.duration") from exc
        if duration <= 0.0:
            raise ValueError(f"{path}: metadata.duration must be positive")
        sample_id = normalize_sample_id(motion.name)
        input_payload = _require_mapping(payload, "input", path)
        modalities = _require_mapping(input_payload, "modalities", path)
        records.append(TaskRecord(
            sample_id=sample_id,
            task_id=task_id,
            task_level=str(metadata.get("task_level", "")),
            task_family=str(metadata.get("task_family", "")),
            task_type=str(metadata.get("task_type", "")),
            duration_seconds=duration,
            task_json=path,
            motion_groundtruth=motion,
            modalities=dict(modalities),
        ))
    if not records:
        raise RuntimeError(f"no task JSON files found under {task_root}")
    return records


def select_unique_base_records(records: Iterable[TaskRecord]) -> dict[str, TaskRecord]:
    """Index Level-1 records, rejecting ambiguous duplicate base IDs."""
    result: dict[str, TaskRecord] = {}
    for record in records:
        previous = result.get(record.sample_id)
        if previous is not None and previous.task_json != record.task_json:
            raise ValueError(
                f"ambiguous base task for {record.sample_id}: "
                f"{previous.task_json} and {record.task_json}"
            )
        result[record.sample_id] = record
    return result


def resolve_condition(record: TaskRecord, modality: str, dataset_root: Path) -> ConditionSpec:
    """Resolve the original condition for one Level-1 task record."""
    try:
        encoder, field = _MODALITY_FIELDS[modality]
    except KeyError as exc:
        raise ValueError(f"unsupported modality {modality!r}; choose from {supported_modalities()}") from exc
    raw = record.modalities.get(field)
    if modality == "text":
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{record.task_json}: input.modalities.text must be non-empty")
        return ConditionSpec(modality, encoder, field, raw.strip(), ())
    values = _path_values(raw, record.task_json, field)
    paths = tuple(_resolve_dataset_path(value, dataset_root, record.task_json) for value in values)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{record.task_json}: condition asset does not exist: {path}")
    if modality in {"audio", "rhythm", "trajectory"} and len(paths) != 1:
        raise ValueError(f"{record.task_json}: {modality} requires exactly one {field} asset")
    return ConditionSpec(modality, encoder, field, None, paths)


def build_instruction_manifest(
    records: Iterable[TaskRecord], *, modality: str, dataset_root: Path, destination: Path
) -> Path:
    """Write an existing evaluator-compatible manifest for resolved conditions."""
    selected = sorted(records, key=lambda item: item.sample_id)
    if not selected:
        raise ValueError("cannot materialize an instruction manifest for zero records")
    condition_rows = [(record, resolve_condition(record, modality, dataset_root)) for record in selected]
    encoder = condition_rows[0][1].encoder
    if any(condition.encoder != encoder for _, condition in condition_rows):
        raise RuntimeError("condition encoder mismatch within one manifest")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if modality == "text":
        payload: dict[str, Any] = {
            "schema": "liujiahui.instruction_manifest.v1",
            "kind": "text",
            "encoder": encoder,
            "format": "inline_text_v1",
            "samples": [
                {"sample_id": record.sample_id, "text": condition.text}
                for record, condition in condition_rows
            ],
        }
    else:
        samples: list[dict[str, Any]] = []
        for record, condition in condition_rows:
            first = condition.paths[0]
            row: dict[str, Any] = {"sample_id": record.sample_id, "path": str(first)}
            if modality in {"audio", "rhythm"}:
                row["assets"] = {"raw_audio": str(first)}
            elif modality in {"human_video", "skeleton_video"}:
                row["assets"] = {"video": [str(path) for path in condition.paths]}
            samples.append(row)
        payload = {
            "schema": "liujiahui.instruction_manifest.v1",
            "kind": modality,
            "encoder": encoder,
            "format": "resolved_assets_v1",
            "samples": samples,
        }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def build_timing_manifest(records: Iterable[TaskRecord], destination: Path) -> Path:
    """Write compact, authoritative task-duration JSONL for the conventional runner."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"sample_id": record.sample_id, "duration": record.duration_seconds}, ensure_ascii=False)
        for record in sorted(records, key=lambda item: item.sample_id)
    ]
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination


def materialize_motion_root(samples: dict[str, Path], destination: Path) -> Path:
    """Create a narrow symlink view containing only selected complete motions."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for sample_id, source in sorted(samples.items()):
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"motion source does not exist: {source}")
        target = destination / sample_id
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                raise FileExistsError(
                    f"refusing to replace existing materialized motion path {target}; "
                    "choose an empty output directory"
                )
            continue
        target.symlink_to(source, target_is_directory=True)
    return destination


def discover_motion_clips(root: Path) -> dict[str, Path]:
    """Return raw clip-directory names, preserving Level-2 constraint suffixes."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"prediction root does not exist: {root}")
    clips: dict[str, Path] = {}
    for joint_csv in root.rglob("joint_pos.csv"):
        clip = joint_csv.parent
        if not all((clip / filename).is_file() for filename in REQUIRED_CSV):
            continue
        raw_name = clip.parent.name if clip.name == "sonic" else clip.name
        previous = clips.get(raw_name)
        if previous is not None and previous != clip:
            raise ValueError(f"duplicate raw prediction clip name {raw_name!r}: {previous}, {clip}")
        clips[raw_name] = clip
    if not clips:
        raise RuntimeError(f"no complete prediction clips found under {root}")
    return clips


def match_level2_prediction_clips(
    prediction_root: Path, records: Iterable[TaskRecord], *, allow_unmatched: bool = False
) -> dict[str, tuple[TaskRecord, Path]]:
    """Match constrained rollout names to Level-2 task JSON records.

    The task JSON is authoritative.  A rollout must end with
    ``<base_sample_id>_<metadata.task_type>`` (optionally followed by the
    conventional fixed-length ``_120`` marker), which handles source IDs that
    themselves contain underscores without brittle generic splitting.
    """
    candidates = discover_motion_clips(prediction_root)
    by_suffix: dict[str, TaskRecord] = {}
    by_source: dict[str, list[TaskRecord]] = {}
    for record in records:
        if not record.task_type:
            raise ValueError(f"{record.task_json}: Level-2 task_type must be non-empty")
        suffix = f"{record.sample_id}_{record.task_type}"
        previous = by_suffix.get(suffix)
        if previous is not None:
            raise ValueError(
                f"ambiguous Level-2 suffix {suffix!r}: {previous.task_json}, {record.task_json}"
            )
        by_suffix[suffix] = record
        by_source.setdefault(record.sample_id, []).append(record)
    matched: dict[str, tuple[TaskRecord, Path]] = {}
    unmatched: list[str] = []
    for raw_name, clip in sorted(candidates.items()):
        name = raw_name.removeprefix("res_")
        # UH-1 appends _continuous to Level-2 rollouts. Try both forms because
        # Trajectory task types can themselves end in "continuous".
        variants = [name]
        if name.endswith("_continuous"):
            variants.append(name.removesuffix("_continuous"))
        match_names = []
        for variant in variants:
            # Body Restrain source IDs may legitimately end in _120, while
            # other exporters use _120 as a fixed-length rollout marker.
            if variant.startswith("L2_BODYRESTRAIN_TXT_"):
                match_names.append(variant)
                if variant.endswith("_120"):
                    match_names.append(variant.removesuffix("_120"))
            else:
                match_names.append(variant.removesuffix("_120"))
        hits_by_id = {
            record.task_id: record
            for match_name in match_names
            for suffix, record in by_suffix.items()
            if match_name.endswith(suffix)
        }
        hits = list(hits_by_id.values())
        # GEM Video exports source IDs without the task-type suffix. Direction
        # has two tasks (left/right) for the same source rollout; score both.
        if not hits:
            source_hits = {record.task_id: record for match_name in match_names
                           for record in by_source.get(match_name, [])}
            hits = list(source_hits.values())
        # Existing Body Restrain exporters sometimes prepend a dataset marker.
        if not hits:
            hits = [record for record in by_suffix.values()
                    if "".join(character for character in record.task_family.casefold() if character.isalnum()) == "bodyrestrain"
                    and any(match_name.endswith(record.sample_id) for match_name in match_names)]
        direction_pair = (len(hits) == 2 and
                          all(record.task_family.casefold() == "direction" for record in hits) and
                          len({record.sample_id for record in hits}) == 1 and
                          len({record.task_type for record in hits}) == 2)
        if len(hits) != 1 and not direction_pair:
            unmatched.append(raw_name)
            continue
        for record in hits:
            if record.task_id in matched:
                raise ValueError(f"duplicate Level-2 rollout for task {record.task_id}: {matched[record.task_id][1]}, {clip}")
            matched[record.task_id] = (record, clip)
    if unmatched and not allow_unmatched:
        preview = ", ".join(unmatched[:8])
        suffix = "" if len(unmatched) <= 8 else f", … ({len(unmatched)} total)"
        raise ValueError(f"Level-2 rollouts do not match any task JSON: {preview}{suffix}")
    if not matched:
        raise RuntimeError("no Level-2 prediction clips matched the selected task JSON")
    return matched


def _read_task_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid task JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: task JSON must be an object")
    return payload


def _require_mapping(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing object {key!r}")
    return value


def _require_string(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: missing non-empty string {key!r}")
    return value


def _resolve_dataset_path(value: str, dataset_root: Path, task_json: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _path_values(value: Any, task_json: Path, field: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = list(value)
    else:
        raise ValueError(f"{task_json}: input.modalities.{field} must be a path or list of paths")
    values = [item.strip() for item in values if item.strip()]
    if not values:
        raise ValueError(f"{task_json}: input.modalities.{field} is empty")
    return values
