from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"Zero root quaternion under {motion_dir}")
    qpos = np.concatenate([root_pos, root_quat / norms, joints], axis=1).astype(np.float32)
    info = parse_info(paths["info"])
    return qpos, {
        "source_files": {key: str(path.resolve()) for key, path in paths.items()},
        "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "source_info": info,
        "csv_quaternion_order": "wxyz",
        "source_frames": int(qpos.shape[0]),
        "source_dimensions": int(qpos.shape[1]),
    }


def _slerp_wxyz(source_times: np.ndarray, quaternions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    left = np.searchsorted(source_times, target_times, side="right") - 1
    left = np.clip(left, 0, len(source_times) - 2)
    right = left + 1
    alpha = ((target_times - source_times[left]) / (source_times[right] - source_times[left]))[:, None]
    first = quaternions[left].copy()
    second = quaternions[right].copy()
    dots = np.sum(first * second, axis=1, keepdims=True)
    second[dots[:, 0] < 0] *= -1.0
    dots = np.abs(dots).clip(-1.0, 1.0)
    close = dots[:, 0] > 0.9995
    output = np.empty_like(first)
    if np.any(close):
        output[close] = (1.0 - alpha[close]) * first[close] + alpha[close] * second[close]
    if np.any(~close):
        theta = np.arccos(dots[~close])
        sine = np.sin(theta)
        output[~close] = (
            np.sin((1.0 - alpha[~close]) * theta) / sine * first[~close]
            + np.sin(alpha[~close] * theta) / sine * second[~close]
        )
    return output / np.linalg.norm(output, axis=1, keepdims=True)


def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos (T,36), got {qpos.shape}")
    if source_fps <= 0 or target_fps <= 0 or len(qpos) < 2:
        raise ValueError("FPS must be positive and at least two frames are required")
    source_times = np.arange(len(qpos), dtype=np.float64) / source_fps
    duration = source_times[-1]
    target_count = int(np.floor(duration * target_fps + 1e-9)) + 1
    target_times = np.minimum(np.arange(target_count, dtype=np.float64) / target_fps, duration)
    output = np.empty((target_count, 36), dtype=np.float64)
    for column in list(range(3)) + list(range(7, 36)):
        output[:, column] = np.interp(target_times, source_times, qpos[:, column])
    quaternions = qpos[:, 3:7]
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    output[:, 3:7] = _slerp_wxyz(source_times, quaternions, target_times)
    return output.astype(np.float32), target_times.astype(np.float32)


def make_windows(values: np.ndarray, window_frames: int, stride: int) -> tuple[np.ndarray, np.ndarray, list[WindowRecord]]:
    if window_frames <= 0 or stride <= 0 or len(values) == 0:
        raise ValueError("Window/stride must be positive and input must be non-empty")
    frames = len(values)
    starts = [0] if frames <= window_frames else list(range(0, frames - window_frames + 1, stride))
    if frames > window_frames and starts[-1] != frames - window_frames:
        starts.append(frames - window_frames)
    windows, masks, records = [], [], []
    for index, start in enumerate(starts):
        chunk = values[start : start + window_frames]
        valid = len(chunk)
        padded = window_frames - valid
        if padded:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], padded, axis=0)], axis=0)
        windows.append(chunk)
        masks.append(np.arange(window_frames) < valid)
        records.append(WindowRecord(index, start, min(start + window_frames, frames), valid, padded))
    return np.stack(windows), np.stack(masks), records
