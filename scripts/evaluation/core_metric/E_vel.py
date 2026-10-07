"""E_vel adapter using OMG's canonical velocity-error implementation."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from scripts.evaluation.core_metric.g_MPJPE import GlobalMPJPEEvaluator, OMG_SOURCE_ROOT, strict_omg_sonic_tracking_qpos


@dataclass(frozen=True)
class VelocityErrorScore:
    """One matched motion pair's E_vel score in millimetres per frame."""

    value: float
    num_frames: int
    num_velocity_frames: int
    num_joints: int
    duration_seconds: float
    groundtruth_hold_frames: int


class VelocityErrorEvaluator:
    """Native-50-Hz G1 FK wrapper around ``omg.benchmarks.metrics.e_vel``.

    The metric itself is imported from OMG rather than reimplemented here, so
    its definition remains exactly ``mean(||diff(pred_pos)-diff(gt_pos)||)``
    in mm/frame. It follows Video/SONIC tracking: GT hold removal, then strict
    50-Hz frame equality; only the official 14-link body subset is evaluated.
    """

    def __init__(self, *, device: str = "auto") -> None:
        self._fk = GlobalMPJPEEvaluator(device=device)
        if str(OMG_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(OMG_SOURCE_ROOT))
        from omg.benchmarks.metrics.tracking import e_vel

        self._e_vel = e_vel

    @property
    def body_order(self) -> tuple[str, ...]:
        return self._fk.body_order

    def score(
        self,
        prediction_qpos_36: np.ndarray,
        groundtruth_qpos_36: np.ndarray,
        *,
        prediction_fps: float,
        groundtruth_fps: float,
        target_fps: float,
        groundtruth_hold_frames: int,
    ) -> VelocityErrorScore:
        prediction, groundtruth, duration = strict_omg_sonic_tracking_qpos(
            prediction_qpos_36,
            groundtruth_qpos_36,
            prediction_fps=prediction_fps,
            groundtruth_fps=groundtruth_fps,
            expected_fps=target_fps,
            groundtruth_hold_frames=groundtruth_hold_frames,
        )
        torch = self._fk.torch
        with torch.inference_mode():
            qpos = torch.as_tensor(
                np.stack((prediction, groundtruth)), dtype=torch.float32, device=self._fk.device
            )
            positions = self._fk.kinematics.forward_kinematics(qpos)["body_pos_w"]
            body_indices = torch.as_tensor(self._fk.body_indices, device=self._fk.device)
            positions = positions.index_select(dim=2, index=body_indices)
            positions_np = positions.detach().cpu().numpy()
        value = self._e_vel(positions_np[0], positions_np[1])
        return VelocityErrorScore(
            value=float(value),
            num_frames=int(positions_np.shape[1]),
            num_velocity_frames=int(positions_np.shape[1] - 1),
            num_joints=int(positions_np.shape[2]),
            duration_seconds=duration,
            groundtruth_hold_frames=int(groundtruth_hold_frames),
        )
