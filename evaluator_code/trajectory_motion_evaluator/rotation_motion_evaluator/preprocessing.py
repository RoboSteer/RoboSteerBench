from __future__ import annotations

import hashlib
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .runtime import load_qpos36, make_windows, resample_qpos, sha256_file

if TYPE_CHECKING:
    from .config import RotationProjectConfig


# retargeted_motion.dof_pos / MUL_POS pack order -> converted_motion joint_pos.csv
# order. Verified numerically on both evaluator training data and benchmark data.
RETARGETED_TO_CONVERTED_DOF = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


class _Numpy2CompatibleUnpickler(pickle.Unpickler):
    """Read NumPy 2 array pickles in the NumPy 1 evaluation environment."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _normalise_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Quaternion contains a zero-norm value")
    quaternion = quaternion / norm
    # q and -q are the same rotation. Enforce a deterministic hemisphere.
    quaternion[quaternion[:, 0] < 0.0] *= -1.0
    return quaternion


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    output = np.asarray(quaternion, dtype=np.float64).copy()
    output[..., 1:] *= -1.0
    return output / np.sum(np.square(output), axis=-1, keepdims=True)


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = _normalise_quaternion_wxyz(quaternion)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def _rotate_vectors_wxyz(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = _normalise_quaternion_wxyz(np.asarray(quaternion).reshape(-1, 4))[0]
    vectors = np.asarray(vectors, dtype=np.float64)
    w = quaternion[0]
    x, y, z = quaternion[1:]
    vx, vy, vz = np.moveaxis(vectors, -1, 0)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.stack(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        axis=-1,
    )


def load_rotation_condition(path: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    with path.open("rb") as stream:
        values = _Numpy2CompatibleUnpickler(stream).load()
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"Missing retargeted-motion fields in {path}: {missing}")
    fps = float(values["fps"])
    root_pos = np.asarray(values["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(values["root_rot"], dtype=np.float64)
    dof = np.asarray(values["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos (T,3), got {root_pos.shape} in {path}")
    if root_xyzw.ndim != 2 or root_xyzw.shape[1] != 4:
        raise ValueError(f"Expected root_rot (T,4), got {root_xyzw.shape} in {path}")
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"Expected dof_pos (T,29), got {dof.shape} in {path}")
    if not (len(root_pos) == len(root_xyzw) == len(dof)):
        raise ValueError(f"Condition frame mismatch in {path}")
    root_wxyz = _normalise_quaternion_wxyz(root_xyzw[:, [3, 0, 1, 2]])
    dof_converted_order = dof[:, RETARGETED_TO_CONVERTED_DOF]
    qpos = np.concatenate([root_pos, root_wxyz, dof_converted_order], axis=1).astype(np.float32)
    return qpos, fps, {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_fps": fps,
        "source_frames": len(qpos),
        "source_quaternion_order": "xyzw",
        "derived_quaternion_order": "wxyz",
        "source_dof_order": "retargeted_motion.dof_pos / MUL_POS pack order",
        "derived_dof_order": "converted_motion joint_pos.csv order",
        "dof_permutation": RETARGETED_TO_CONVERTED_DOF.tolist(),
    }


def load_rotation_target(motion_dir: Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    qpos, metadata = load_qpos36(motion_dir)
    info = metadata["source_info"]
    source_fps = float(info.get("target_fps", 0))
    hold_frames = int(info.get("start_hold_frames", 0))
    if source_fps <= 0:
        raise ValueError(f"Missing target_fps in {motion_dir / 'info.txt'}")
    if hold_frames < 0 or hold_frames >= len(qpos):
        raise ValueError(f"Invalid start_hold_frames={hold_frames} for {len(qpos)} frames in {motion_dir}")
    without_hold = qpos[hold_frames:].copy()
    return without_hold, source_fps, {
        **metadata,
        "input_fps": source_fps,
        "input_frames_with_hold": len(qpos),
        "removed_start_hold_frames": hold_frames,
        "frames_after_hold_removal": len(without_hold),
    }


def qpos_to_rotation_features(qpos_wxyz: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos_wxyz, dtype=np.float64)
    root_position = qpos[:, :3]
    root_quaternion = _normalise_quaternion_wxyz(qpos[:, 3:7])
    joint_angles = qpos[:, 7:36]
    anchor_inverse = _quat_inverse_wxyz(root_quaternion[:1])
    relative_position = _rotate_vectors_wxyz(anchor_inverse, root_position - root_position[:1])
    relative_quaternion = _quat_multiply_wxyz(np.repeat(anchor_inverse, len(qpos), axis=0), root_quaternion)
    matrix = _quat_to_matrix_wxyz(relative_quaternion)
    rotation_6d = matrix[:, :, :2].reshape(len(qpos), 6)
    # Fixed physical scales avoid fitting normalisation statistics on benchmark data.
    features = np.concatenate([relative_position / 2.0, rotation_6d, joint_angles / np.pi], axis=1)
    if features.shape[1] != 38 or not np.isfinite(features).all():
        raise ValueError(f"Invalid rotation feature shape/content: {features.shape}")
    return features.astype(np.float32)


def kinematic_fingerprint(features: np.ndarray) -> str:
    quantised = np.round(np.asarray(features, dtype=np.float32), decimals=4)
    return hashlib.sha256(quantised.tobytes(order="C")).hexdigest()


def derive_pair_arrays(
    retargeted_path: Path,
    motion_dir: Path,
    config: RotationProjectConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    condition_qpos, condition_fps, condition_metadata = load_rotation_condition(retargeted_path)
    motion_qpos, motion_fps, motion_metadata = load_rotation_target(motion_dir)
    condition_30, condition_times = resample_qpos(condition_qpos, condition_fps, config.derived_fps)
    motion_30, motion_times = resample_qpos(motion_qpos, motion_fps, config.derived_fps)
    condition_features = qpos_to_rotation_features(condition_30)
    motion_features = qpos_to_rotation_features(motion_30)
    condition_qpos_windows, condition_masks, condition_window_records = make_windows(
        condition_30, config.window_frames, config.window_stride
    )
    motion_qpos_windows, motion_masks, motion_window_records = make_windows(
        motion_30, config.window_frames, config.window_stride
    )
    condition_feature_windows, _, _ = make_windows(
        condition_features, config.window_frames, config.window_stride
    )
    motion_feature_windows, _, _ = make_windows(motion_features, config.window_frames, config.window_stride)
    arrays = {
        "condition_qpos_30fps": condition_30,
        "motion_qpos_30fps": motion_30,
        "condition_frame_times": condition_times.astype(np.float32),
        "motion_frame_times": motion_times.astype(np.float32),
        "condition_rotation_windows": condition_feature_windows,
        "motion_rotation_windows": motion_feature_windows,
        "condition_window_masks": condition_masks,
        "motion_window_masks": motion_masks,
        "condition_qpos_windows": condition_qpos_windows,
        "motion_qpos_windows": motion_qpos_windows,
    }
    metadata = {
        "condition": condition_metadata,
        "motion": motion_metadata,
        "derived_fps": config.derived_fps,
        "window_policy": {
            "window_frames": config.window_frames,
            "stride_frames": config.window_stride,
            "tail_policy": "append_final_full_window",
            "short_motion_padding": "repeat_last_frame_masked_invalid",
            "aggregation": "mean_window_embeddings_then_l2_normalize",
        },
        "condition_derived_frames": len(condition_30),
        "motion_derived_frames": len(motion_30),
        "condition_windows": [asdict(item) for item in condition_window_records],
        "motion_windows": [asdict(item) for item in motion_window_records],
        "condition_kinematic_id": kinematic_fingerprint(condition_features),
        "quaternion_and_root_policy": {
            "condition_quaternion": "xyzw_to_wxyz_then_unit_normalize_and_positive_w_hemisphere",
            "motion_quaternion": "CSV wxyz unit normalized during qpos loading",
            "root_position": "relative_to_first_frame_then_rotate_by_inverse_first_root_rotation",
            "root_rotation": "relative_to_first_frame_then_first_two_rotation_matrix_columns",
            "joint_angle_scale": "divide_by_pi",
            "root_position_scale_meters": "divide_by_2",
        },
    }
    return arrays, metadata
