from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audio_motion_evaluator.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from audio_motion_evaluator.metrics import semantic_aware_group_order


EXTERNAL_SCHEMA = "omg.external.sample.v1"
TASK_PREFIX = "L1_MUL_AUD_"
GROUP_SIZE = 32


def _load_csv(path: Path, widths: tuple[int, ...]) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if data.ndim != 2 or data.shape[1] not in widths:
        raise ValueError(f"{path}: expected column width in {widths}, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: contains non-finite values")
    return data


def _motion_files(clip_dir: Path) -> dict[str, Path]:
    return {
        "joint": clip_dir / "joint_pos.csv",
        "root_pos": clip_dir / "body_pos.csv",
        "root_quat": clip_dir / "body_quat.csv",
        "info": clip_dir / "info.txt",
    }


def _read_fps(info_path: Path) -> float:
    if not info_path.is_file():
        raise FileNotFoundError(f"Motion FPS metadata is missing: {info_path}")
    values: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    raw = values.get("target_fps") or values.get("source_fps")
    if not raw or float(raw) <= 0:
        raise ValueError(f"Could not determine positive FPS from {info_path}")
    return float(raw)


def _load_qpos(clip_dir: Path) -> tuple[np.ndarray, float, dict[str, str]]:
    files = _motion_files(clip_dir)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing motion files under {clip_dir}: {missing}")
    joint = _load_csv(files["joint"], (29,))
    root_pos = _load_csv(files["root_pos"], (3, 90))[:, :3]
    root_quat = _load_csv(files["root_quat"], (4, 120))[:, :4]
    frames = min(len(joint), len(root_pos), len(root_quat))
    if frames < 2:
        raise ValueError(f"{clip_dir}: motion has fewer than two frames")
    root_quat = root_quat[:frames]
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"{clip_dir}: invalid root quaternion")
    root_quat = root_quat / norms
    root_quat[root_quat[:, 0] < 0] *= -1
    qpos = np.concatenate((root_pos[:frames], root_quat, joint[:frames]), axis=1).astype(np.float32)
    hashes = {key: sha256_file(path) for key, path in files.items()}
    return qpos, _read_fps(files["info"]), hashes


def _task_id_from_dir_name(name: str) -> str:
    return name if name.startswith(TASK_PREFIX) else f"{TASK_PREFIX}{name}"


def _discover_generated(root: Path) -> dict[str, Path]:
    """Inspect only root and one category level; never recurse through unrelated trees."""
    found: dict[str, Path] = {}
    top_level = sorted(path for path in root.iterdir() if path.is_dir())
    clip_dirs = [path for path in top_level if (_motion_files(path)["joint"]).is_file()]
    category_dirs = [path for path in top_level if not (_motion_files(path)["joint"]).is_file()]
    for category in category_dirs:
        clip_dirs.extend(
            sorted(
                path
                for path in category.iterdir()
                if path.is_dir() and (_motion_files(path)["joint"]).is_file()
            )
        )
    for clip_dir in clip_dirs:
        if not (_motion_files(clip_dir)["joint"]).is_file():
            continue
        task_id = _task_id_from_dir_name(clip_dir.name)
        if task_id in found and found[task_id] != clip_dir:
            raise RuntimeError(f"Duplicate generated task {task_id}: {found[task_id]} and {clip_dir}")
        found[task_id] = clip_dir
    return found


def _resolve_workspace_path(workspace_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _caption_text(path: Path) -> str:
    value = read_json(path)
    dense = value.get("dense_detail") or {}
    text = str(dense.get("full_sequence") or value.get("macro_summary") or "").strip()
    if not text:
        raise ValueError(f"No full_sequence or macro_summary in {path}")
    return text


def _semantic_id(text: str) -> str:
    normalised = " ".join(text.strip().lower().split())
    return hashlib.sha256(f"semantic-text-v1\0{normalised}".encode("utf-8")).hexdigest()


def _cache_paths(output_dir: Path, task_id: str) -> tuple[Path, Path]:
    suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[-100:]
    return (
        output_dir / "pair_cache" / f"{stem}__{suffix}.npz",
        output_dir / "pair_metadata" / f"{stem}__{suffix}.json",
    )


def _load_cached_pair(npz_path: Path, metadata_path: Path, task_id: str) -> dict[str, Any] | None:
    if not npz_path.is_file() or not metadata_path.is_file():
        if npz_path.exists() != metadata_path.exists():
            raise RuntimeError(f"Incomplete pair checkpoint: {npz_path}, {metadata_path}")
        return None
    metadata = read_json(metadata_path)
    if metadata.get("task_id") != task_id or metadata.get("pair_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"Pair checkpoint identity/hash mismatch: {npz_path}")
    source_files = metadata.get("source_files") or {}
    source_hashes = metadata.get("source_sha256") or {}
    for key in ("task", "audio", "caption"):
        path = Path(str(source_files.get(key, "")))
        if not path.is_file() or sha256_file(path) != source_hashes.get(key):
            raise RuntimeError(f"Cached pair source changed for {task_id}: {key}={path}")
    for label, source_key, hash_key in (
        ("generated", "generated_motion", "generated_motion_files"),
        ("reference", "reference_motion", "reference_motion_files"),
    ):
        directory = Path(str(source_files.get(source_key, "")))
        expected_hashes = source_hashes.get(hash_key) or {}
        for file_key, file_path in _motion_files(directory).items():
            if not file_path.is_file() or sha256_file(file_path) != expected_hashes.get(file_key):
                raise RuntimeError(
                    f"Cached pair {label} source changed for {task_id}: {file_path}"
                )
    with np.load(npz_path, allow_pickle=False) as handle:
        return {
            "task_id": str(handle["task_id"]),
            "caption": str(handle["caption"]),
            "semantic_id": str(handle["semantic_id"]),
            "audio_path": str(handle["audio_path"]),
            "generated_path": str(handle["generated_path"]),
            "gt_path": str(handle["gt_path"]),
            "task_path": str(handle["task_path"]),
            "caption_path": str(handle["caption_path"]),
            "duration": float(handle["duration"]),
            "generated_qpos": handle["generated_qpos"].astype(np.float32, copy=True),
            "reference_qpos": handle["reference_qpos"].astype(np.float32, copy=True),
            "generated_fps": float(handle["generated_fps"]),
            "reference_fps": float(handle["reference_fps"]),
        }


def _derive_pair(
    task_id: str,
    generated_path: Path,
    task_root: Path,
    workspace_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], bool]:
    cache_path, metadata_path = _cache_paths(output_dir, task_id)
    cached = _load_cached_pair(cache_path, metadata_path, task_id)
    if cached is not None:
        return cached, True
    task_path = task_root / f"{task_id}.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"Task JSON is missing: {task_path}")
    task = read_json(task_path)
    if str(task.get("metadata", {}).get("task_type")) != "MUL_AUD":
        raise ValueError(f"Unexpected task_type in {task_path}")
    audio_values = task.get("input", {}).get("modalities", {}).get("audios") or []
    if len(audio_values) != 1:
        raise ValueError(f"Expected exactly one Audio input in {task_path}, got {audio_values}")
    audio_path = _resolve_workspace_path(workspace_root, audio_values[0])
    gt_path = _resolve_workspace_path(workspace_root, task["ground_truth"]["motion_parameters"])
    source_id = task_id[len(TASK_PREFIX) :]
    caption_path = workspace_root / "Data_Processed" / "polished_captions" / f"{source_id}.json"
    if not audio_path.is_file() or not gt_path.is_dir() or not caption_path.is_file():
        raise FileNotFoundError(
            f"Incomplete task assets for {task_id}: audio={audio_path.is_file()} "
            f"gt={gt_path.is_dir()} caption={caption_path.is_file()}"
        )
    caption = _caption_text(caption_path)
    generated_qpos, generated_fps, generated_hashes = _load_qpos(generated_path)
    reference_qpos, reference_fps, reference_hashes = _load_qpos(gt_path)
    pair = {
        "task_id": task_id,
        "caption": caption,
        "semantic_id": _semantic_id(caption),
        "audio_path": str(audio_path),
        "generated_path": str(generated_path.resolve()),
        "gt_path": str(gt_path),
        "task_path": str(task_path.resolve()),
        "caption_path": str(caption_path.resolve()),
        "duration": float(task.get("metadata", {}).get("duration") or 0.0),
        "generated_qpos": generated_qpos,
        "reference_qpos": reference_qpos,
        "generated_fps": generated_fps,
        "reference_fps": reference_fps,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{key: np.asarray(value) for key, value in pair.items()},
        )
    os.replace(temporary, cache_path)
    metadata = {
        "schema": "mul_aud_omg_pair_cache.v1",
        "task_id": task_id,
        "source_files": {
            "task": str(task_path.resolve()),
            "audio": str(audio_path),
            "caption": str(caption_path.resolve()),
            "generated_motion": str(generated_path.resolve()),
            "reference_motion": str(gt_path),
        },
        "source_sha256": {
            "task": sha256_file(task_path),
            "audio": sha256_file(audio_path),
            "caption": sha256_file(caption_path),
            "generated_motion_files": generated_hashes,
            "reference_motion_files": reference_hashes,
        },
        "generated_fps": generated_fps,
        "generated_frames": len(generated_qpos),
        "reference_fps": reference_fps,
        "reference_frames": len(reference_qpos),
        "pair_npz": str(cache_path.resolve()),
        "pair_npz_sha256": sha256_file(cache_path),
    }
    atomic_write_json(metadata_path, metadata)
    return pair, False


def _padded_stack(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray([len(value) for value in values], dtype=np.int32)
    maximum = int(lengths.max())
    output = np.empty((len(values), maximum, 36), dtype=np.float32)
    valid = np.zeros((len(values), maximum), dtype=np.bool_)
    for index, value in enumerate(values):
        length = len(value)
        output[index, :length] = value
        output[index, length:] = value[-1]
        valid[index, :length] = True
    return output, lengths, valid


def _write_npz_atomic(path: Path, **values: Any) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MUL_AUD generated/GT Motion and speech as an OMG semantic artifact.")
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=SCRIPT_DIR.parents[1])
    parser.add_argument(
        "--task-root",
        type=Path,
        default=SCRIPT_DIR.parents[1]
        / "Tasks"
        / "Level1"
        / "full_conditioning_reproduction"
        / "audio_to_motion_generation"
        / "MUL_AUD",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    generated_root = args.generated_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not generated_root.is_dir() or not task_root.is_dir():
        raise FileNotFoundError(f"Missing generated/task root: {generated_root}, {task_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = _discover_generated(generated_root)
    if not generated:
        raise RuntimeError(f"No generated Motion CSV directories found under {generated_root} (max depth: 2)")
    print(f"[discover] generated_motion={len(generated)} task_root={task_root}", flush=True)
    pairs: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    cached_count = 0
    for task_id, generated_path in tqdm(sorted(generated.items()), desc="prepare MUL_AUD", unit="pair"):
        try:
            pair, cached = _derive_pair(task_id, generated_path, task_root, workspace_root, output_dir)
            pairs.append(pair)
            cached_count += int(cached)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"task_id": task_id, "generated_path": str(generated_path), "error": str(exc)})
    if len(pairs) < GROUP_SIZE:
        atomic_write_json(output_dir / "invalid_pairs.json", invalid)
        raise RuntimeError(f"Only {len(pairs)} valid pairs; fixed-batch32 evaluation needs at least {GROUP_SIZE}")
    semantic_ids = [pair["semantic_id"] for pair in pairs]
    order, excluded = semantic_aware_group_order(
        semantic_ids, GROUP_SIZE, args.seed, dataset_names=["MUL_AUD"] * len(pairs)
    )
    generated_qpos, generated_lengths, generated_valid = _padded_stack(
        [pair["generated_qpos"] for pair in pairs]
    )
    reference_qpos, reference_lengths, reference_valid = _padded_stack(
        [pair["reference_qpos"] for pair in pairs]
    )
    captions = np.asarray([pair["caption"] for pair in pairs], dtype=np.str_)
    datasets = np.full(len(pairs), "MUL_AUD", dtype=np.str_)
    indices = np.arange(len(pairs), dtype=np.int32)
    _write_npz_atomic(
        output_dir / "generated_qpos.npz",
        qpos_36=generated_qpos,
        valid_lengths=generated_lengths,
        valid=generated_valid,
        fps=np.asarray([pair["generated_fps"] for pair in pairs], dtype=np.float32),
        captions=captions,
        dataset=datasets,
        dataset_index=indices,
    )
    _write_npz_atomic(
        output_dir / "reference_qpos.npz",
        qpos_36=reference_qpos,
        valid_lengths=reference_lengths,
        valid=reference_valid,
        fps=np.asarray([pair["reference_fps"] for pair in pairs], dtype=np.float32),
        captions=captions,
        dataset=datasets,
        dataset_index=indices,
    )
    sample_rows = [
        {
            "schema": EXTERNAL_SCHEMA,
            "dataset": "MUL_AUD",
            "index": index,
            "caption": pair["caption"],
            "semantic_id": pair["semantic_id"],
            "source_id": pair["task_id"],
            "meta": {
                "audio_path": pair["audio_path"],
                "semantic_id": pair["semantic_id"],
                "caption_path": pair["caption_path"],
                "generated_path": pair["generated_path"],
                "gt_path": pair["gt_path"],
                "task_path": pair["task_path"],
                "target_duration_seconds": pair["duration"],
                "motion_alignment": "full generated/reference clips; padded storage with valid_lengths; evaluator crops before 30-FPS derivation",
            },
        }
        for index, pair in enumerate(pairs)
    ]
    atomic_write_jsonl(output_dir / "samples.jsonl", sample_rows)
    atomic_write_json(output_dir / "invalid_pairs.json", invalid)
    report = {
        "schema": "mul_aud_omg_artifact_report.v1",
        "generated_root": str(generated_root),
        "task_root": str(task_root),
        "valid_pairs": len(pairs),
        "cached_pairs": cached_count,
        "invalid_pairs": len(invalid),
        "retrieval_group_size": GROUP_SIZE,
        "retrieval_seed": args.seed,
        "retrieval_groups": len(order) // GROUP_SIZE,
        "retrieval_pairs": len(order),
        "excluded_remainder": len(excluded),
        "excluded_task_ids": [pairs[int(index)]["task_id"] for index in excluded],
        "semantic_conflicts": 0,
        "generated_padded_frames": int(generated_qpos.shape[1]),
        "reference_padded_frames": int(reference_qpos.shape[1]),
        "storage_policy": "repeat-last padding plus valid_lengths/valid mask; padding is never encoded",
    }
    atomic_write_json(output_dir / "match_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
