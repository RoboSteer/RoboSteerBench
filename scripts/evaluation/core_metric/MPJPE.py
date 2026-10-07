"""Root-translation-aligned MPJPE for G1 motions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.evaluation.core_metric.g_MPJPE import GlobalMPJPEEvaluator, strict_omg_sonic_tracking_qpos


@dataclass(frozen=True)
class MPJPEScore:
    """One complete prediction--GT motion pair's root-aligned MPJPE, in mm."""

    value: float
    num_frames: int
    num_joints: int
    duration_seconds: float
    groundtruth_hold_frames: int


class RootAlignedMPJPEEvaluator(GlobalMPJPEEvaluator):
    """G1 FK adapter for the standard pelvis-translation-aligned MPJPE.

    At every frame, the world position of the G1 pelvis/root link is subtracted
    independently from prediction and reference body-link positions.  This
    removes global translation only: global root orientation, articulation,
    scale, and all other rigid alignment remain unmodified.
    """

    root_link_name = "pelvis"

    def __init__(self, *, device: str = "auto") -> None:
        super().__init__(device=device)
        try:
            full_root_link_index = self.body_order.index(self.root_link_name)
            self.root_link_index = self.body_indices.index(full_root_link_index)
        except ValueError as exc:
            raise RuntimeError(
                f"G1 body order does not contain root link {self.root_link_name!r}"
            ) from exc

    def score(
        self,
        prediction_qpos_36: np.ndarray,
        groundtruth_qpos_36: np.ndarray,
        *,
        prediction_fps: float,
        groundtruth_fps: float,
        target_fps: float,
        groundtruth_hold_frames: int,
    ) -> MPJPEScore:
        prediction, groundtruth, duration = strict_omg_sonic_tracking_qpos(
            prediction_qpos_36,
            groundtruth_qpos_36,
            prediction_fps=prediction_fps,
            groundtruth_fps=groundtruth_fps,
            expected_fps=target_fps,
            groundtruth_hold_frames=groundtruth_hold_frames,
        )
        torch = self.torch
        with torch.inference_mode():
            qpos = torch.as_tensor(
                np.stack((prediction, groundtruth)), dtype=torch.float32, device=self.device
            )
            body_pos_w = self.kinematics.forward_kinematics(qpos)["body_pos_w"]
            body_indices = torch.as_tensor(self.body_indices, device=self.device)
            tracked_positions = body_pos_w.index_select(dim=2, index=body_indices)
            root_positions = tracked_positions[:, :, self.root_link_index : self.root_link_index + 1]
            root_aligned = tracked_positions - root_positions
            errors = torch.linalg.vector_norm(root_aligned[0] - root_aligned[1], dim=-1)
        return MPJPEScore(
            value=float(errors.mean().detach().cpu()) * 1000.0,
            num_frames=int(errors.shape[0]),
            num_joints=int(errors.shape[1]),
            duration_seconds=duration,
            groundtruth_hold_frames=int(groundtruth_hold_frames),
        )
