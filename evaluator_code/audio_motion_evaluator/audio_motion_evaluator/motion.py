from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io_utils import atomic_write_json, read_json, safe_stem, sha256_file
from .pairs import PairRecord


@dataclass(frozen=True)
class WindowRecord:
    index: int
    start_frame_30fps: int
    end_frame_exclusive_30fps: int
    valid_frames: int
    padded_frames: int


def parse_info(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line or raw_line.startswith(" "):
            continue
        key, value = (part.strip() for part in raw_line.split(":", 1))
        if key in {"source_fps", "target_fps", "start_hold_frames"}:
            result[key] = int(value)
        elif key == "start_hold_seconds":
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _load_csv(path: Path, expected_columns: int) -> np.ndarray:
    array = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns in {path}, got {array.shape}")
    return array


def load_qpos36(motion_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    paths = {
        "root_position": motion_dir / "body_pos.csv",
        "root_quaternion": motion_dir / "body_quat.csv",
        "joint_position": motion_dir / "joint_pos.csv",
        "info": motion_dir / "info.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing motion inputs: {missing}")
    root_pos = _load_csv(paths["root_position"], 3)
    root_quat = _load_csv(paths["root_quaternion"], 4)
    joints = _load_csv(paths["joint_position"], 29)
    if not (len(root_pos) == len(root_quat) == len(joints)):
        raise ValueError(f"Frame mismatch under {motion_dir}")
    norms = np.sqrt(np.sum(root_quat * root_quat, axis=1))
    if np.any(norms < 1e-8):
        raise ValueError(f"Zero root quaternion under {motion_dir}")
    root_quat = root_quat / norms[:, None]
    qpos = np.concatenate([root_pos, root_quat, joints], axis=1)
    info = parse_info(paths["info"])
    metadata = {
        "source_files": {key: str(path.resolve()) for key, path in paths.items()},
        "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "source_info": info,
        "csv_quaternion_order": "wxyz",
        "source_frames": int(qpos.shape[0]),
        "source_dimensions": int(qpos.shape[1]),
    }
    return qpos, metadata


def _slerp_wxyz(source_times: np.ndarray, quaternions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Shortest-path unit-quaternion SLERP without a SciPy runtime dependency."""
    left = np.searchsorted(source_times, target_times, side="right") - 1
    left = np.clip(left, 0, len(source_times) - 2)
    right = left + 1
    interval = source_times[right] - source_times[left]
    alpha = ((target_times - source_times[left]) / interval)[:, None]
    first = quaternions[left].copy()
    second = quaternions[right].copy()
    dots = np.sum(first * second, axis=1, keepdims=True)
    negative = dots[:, 0] < 0
    second[negative] *= -1.0
    dots = np.abs(dots).clip(-1.0, 1.0)
    close = dots[:, 0] > 0.9995
    output = np.empty_like(first)
    if np.any(close):
        output[close] = (1.0 - alpha[close]) * first[close] + alpha[close] * second[close]
    if np.any(~close):
        theta = np.arccos(dots[~close])
        sine = np.sin(theta)
        weight_first = np.sin((1.0 - alpha[~close]) * theta) / sine
        weight_second = np.sin(alpha[~close] * theta) / sine
        output[~close] = weight_first * first[~close] + weight_second * second[~close]
    return output / np.sqrt(np.sum(output * output, axis=1, keepdims=True))


def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    if len(qpos) < 2:
        raise ValueError("At least two motion frames are required")
    source_times = np.arange(len(qpos), dtype=np.float64) / source_fps
    duration = source_times[-1]
    target_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, duration)
    output = np.empty((target_count, 36), dtype=np.float64)
    for column in list(range(3)) + list(range(7, 36)):
        output[:, column] = np.interp(target_times, source_times, qpos[:, column])
    output[:, 3:7] = _slerp_wxyz(source_times, qpos[:, 3:7], target_times)
    output[:, 3:7] /= np.sqrt(np.sum(output[:, 3:7] * output[:, 3:7], axis=1, keepdims=True))
    return output.astype(np.float32), target_times


def make_windows(qpos: np.ndarray, window_frames: int, stride: int) -> tuple[np.ndarray, np.ndarray, list[WindowRecord]]:
    if window_frames <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive")
    frames = len(qpos)
    if frames <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, frames - window_frames + 1, stride))
        tail = frames - window_frames
        if starts[-1] != tail:
            starts.append(tail)
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    records: list[WindowRecord] = []
    for index, start in enumerate(starts):
        chunk = qpos[start : start + window_frames]
        valid = len(chunk)
        padded = window_frames - valid
        if padded:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], padded, axis=0)], axis=0)
        mask = np.arange(window_frames) < valid
        windows.append(chunk)
        masks.append(mask)
        records.append(WindowRecord(index, start, min(start + window_frames, frames), valid, padded))
    return np.stack(windows), np.stack(masks), records


def derive_motion_pair(pair: PairRecord, config: ProjectConfig) -> tuple[Path, Path, bool]:
    stem = safe_stem(pair.pair_id)
    data_path = config.derived_motion_dir / f"{stem}.npz"
    metadata_path = config.metadata_dir / "motion" / f"{stem}.json"
    if data_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("pair_id") == pair.pair_id and metadata.get("derived_npz_sha256") == sha256_file(data_path):
            return data_path, metadata_path, True
        raise RuntimeError(f"Existing derived motion conflicts with metadata: {data_path}")
    if data_path.exists() != metadata_path.exists():
        raise RuntimeError(f"Incomplete motion checkpoint; inspect before retrying: {data_path}, {metadata_path}")

    qpos, source_metadata = load_qpos36(Path(pair.motion_dir))
    input_fps = float(source_metadata["source_info"].get("target_fps", 0))
    if input_fps <= 0:
        raise ValueError(f"Missing target_fps in {Path(pair.motion_dir) / 'info.txt'}")
    qpos30, target_times = resample_qpos(qpos, input_fps, config.derived_fps)
    windows, masks, records = make_windows(qpos30, config.window_frames, config.window_stride)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            qpos_30fps=qpos30,
            frame_times_seconds=target_times,
            windows_qpos_30fps=windows,
            window_valid_masks=masks,
        )
    os.replace(temporary, data_path)
    metadata = {
        "schema": "audio_motion_derived_motion.v1",
        "pair_id": pair.pair_id,
        **source_metadata,
        "input_fps": input_fps,
        "input_duration_seconds": (len(qpos) - 1) / input_fps,
        "derived_fps": config.derived_fps,
        "derived_frames": int(len(qpos30)),
        "resampling": {
            "time_origin_seconds": 0.0,
            "endpoint_policy": "floor_duration_plus_initial_frame",
            "translation_and_joints": "linear_interpolation",
            "root_quaternion": "shortest_path_unit_quaternion_slerp_wxyz_numpy",
        },
        "window_policy": {
            "window_frames": config.window_frames,
            "stride_frames": config.window_stride,
            "short_motion_padding": "repeat_last_frame_masked_invalid",
            "tail_policy": "append_final_full_window",
            "aggregation": "mean_of_l2_normalized_window_embeddings_then_l2_normalize",
        },
        "windows": [asdict(record) for record in records],
        "derived_npz": str(data_path.resolve()),
        "derived_npz_sha256": sha256_file(data_path),
    }
    atomic_write_json(metadata_path, metadata)
    return data_path, metadata_path, False
