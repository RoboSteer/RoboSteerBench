"""Adapter for the delivered Rotation--Motion evaluator and trajectory conditions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex, InstructionSample
from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator, _resolve_path


# The evaluator was trained with this conversion from public MUL_POS retargeted
# DoF order to the qpos/joint_pos.csv order consumed by OMG.
_RETARGETED_TO_CONVERTED_DOF = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
     24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


class TrajectoryMotionAdapter(CrossModalMotionEvaluator):
    """Use ``RotationMotionEvaluator`` for trajectory--motion MM-Distance."""

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        super().__init__(name, config, registry_root, device)
        delivery_root = _resolve_path(config["root"], registry_root)
        checkpoint = _resolve_path(config["checkpoint"], registry_root)
        omg_root = _resolve_path(config["omg_root"], registry_root)
        omg_checkpoint = _resolve_path(config["omg_checkpoint"], registry_root)
        for label, path, is_dir in (
            ("trajectory evaluator root", delivery_root, True),
            ("trajectory evaluator checkpoint", checkpoint, False),
            ("OMG root", omg_root, True),
            ("OMG motion checkpoint", omg_checkpoint, False),
        ):
            if not (path.is_dir() if is_dir else path.is_file()):
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(delivery_root) not in sys.path:
            sys.path.insert(0, str(delivery_root))
        from rotation_motion_evaluator import RotationMotionEvaluator

        evaluator_device = None if device == "auto" else device
        self._evaluator = RotationMotionEvaluator(
            checkpoint,
            omg_root=omg_root,
            omg_checkpoint=omg_checkpoint,
            device=evaluator_device,
        )
        self._paths = {
            "checkpoint": str(checkpoint),
            "omg_checkpoint": str(omg_checkpoint),
        }

    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del index
        if not sample.path.is_file():
            raise FileNotFoundError(f"trajectory condition does not exist: {sample.path}")
        if sample.path.suffix.lower() == ".pkl":
            return self._evaluator.encode_condition(sample.path)
        if sample.path.suffix.lower() != ".npz":
            raise ValueError(
                f"trajectory_motion requires a .pkl or .npz condition, got {sample.path}"
            )
        with np.load(sample.path, allow_pickle=False) as payload:
            required = ("root_position", "root_rotation", "dof", "fps")
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"trajectory condition {sample.path} is missing {missing}")
            qpos = _condition_arrays_to_qpos(
                payload["root_position"], payload["root_rotation"], payload["dof"]
            )
            source_fps = float(np.asarray(payload["fps"]).item())
        embedding, derived = self._evaluator.encode_qpos(qpos, source_fps)
        return embedding, {
            "source": {
                "sample_id": sample.sample_id,
                "path": str(sample.path),
                "source_fps": source_fps,
                "source_frames": int(len(qpos)),
                "source_quaternion_order": "xyzw",
                "derived_quaternion_order": "wxyz",
                "source_dof_order": "MUL_POS retargeted-motion order",
                "derived_dof_order": "qpos / joint_pos.csv order",
                "dof_permutation": _RETARGETED_TO_CONVERTED_DOF.tolist(),
            },
            "derived": derived,
        }

    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._evaluator.encode_qpos(qpos_36, source_fps)

    def protocol(self) -> dict[str, Any]:
        architecture = self._evaluator.checkpoint_metadata.get("architecture") or {}
        return {
            "evaluator": "RotationMotionEvaluator",
            "condition": "complete_MUL_POS_trajectory_from_raw_pkl_or_preprocessed_npz",
            "condition_input": "root_position + root_rotation_xyzw + retargeted_dof_29",
            "condition_conversion": "xyzw_to_wxyz_and_fixed_retargeted_to_converted_dof_permutation",
            "motion_input": "complete_qpos_36",
            "motion_target_fps": int(self._evaluator.derived_fps),
            "motion_window_frames": int(self._evaluator.window_frames),
            "motion_window_stride": int(self._evaluator.window_stride),
            "motion_aggregation": "mean_window_embeddings_then_l2_normalize",
            "embedding_dim": int(architecture.get("embedding_dim", 512)),
            "l2_normalized": True,
            **self._paths,
        }


def _condition_arrays_to_qpos(
    root_position: np.ndarray, root_rotation_xyzw: np.ndarray, dof: np.ndarray
) -> np.ndarray:
    root_position = np.asarray(root_position, dtype=np.float64)
    root_rotation_xyzw = np.asarray(root_rotation_xyzw, dtype=np.float64)
    dof = np.asarray(dof, dtype=np.float64)
    if root_position.ndim != 2 or root_position.shape[1] != 3:
        raise ValueError(f"expected root_position (T,3), got {root_position.shape}")
    if root_rotation_xyzw.ndim != 2 or root_rotation_xyzw.shape[1] != 4:
        raise ValueError(f"expected root_rotation (T,4), got {root_rotation_xyzw.shape}")
    if dof.ndim != 2 or dof.shape[1] != 29:
        raise ValueError(f"expected dof (T,29), got {dof.shape}")
    if len(root_position) < 2 or not (
        len(root_position) == len(root_rotation_xyzw) == len(dof)
    ):
        raise ValueError("trajectory condition needs at least two aligned frames")
    root_wxyz = root_rotation_xyzw[:, [3, 0, 1, 2]]
    norm = np.linalg.norm(root_wxyz, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("trajectory condition contains a zero-norm root quaternion")
    root_wxyz = root_wxyz / norm
    root_wxyz[root_wxyz[:, 0] < 0.0] *= -1.0
    qpos = np.concatenate(
        [root_position, root_wxyz, dof[:, _RETARGETED_TO_CONVERTED_DOF]], axis=1
    ).astype(np.float32)
    if not np.isfinite(qpos).all():
        raise ValueError("trajectory condition contains non-finite values")
    return qpos
