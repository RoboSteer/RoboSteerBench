"""Contact-period foot sliding metric for G1 qpos motions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.omg_paths import omg_root


OMG_SOURCE_ROOT = omg_root() / "src"
G1_KINEMATICS_PATH = omg_root() / "assets" / "robots" / "g1" / "g1_kinematics.json"


@dataclass(frozen=True)
class ContactSlidingScore:
    """ContactSliding result for one complete source motion."""

    value: float
    num_contact_intervals: int
    num_valid_intervals: int

    @property
    def contact_interval_ratio(self) -> float:
        return 0.0 if self.num_valid_intervals == 0 else self.num_contact_intervals / self.num_valid_intervals


def contact_sliding_from_sole_proxies(
    *, sole_points: Any, sole_radii: Any, fps: Any, contact_height_threshold: float,
    contact_penetration_tolerance: float = 0.02, valid: Any | None = None,
    sole_foot_ids: Any | None = None,
):
    """Return per-motion sliding speeds and contact-interval counts.

    ``sole_points`` has shape ``(B, T, P, 3)``. Invalid padded frames are
    excluded by ``valid`` before any aggregation.
    """
    import torch

    if sole_points.ndim != 4 or sole_points.shape[-1] != 3:
        raise ValueError(f"sole_points must have shape (B, T, P, 3), got {tuple(sole_points.shape)}")
    batch_size, frames, _, _ = sole_points.shape
    if frames < 2:
        raise ValueError("contact sliding requires at least two frames")
    if valid is None:
        valid = torch.ones((batch_size, frames), dtype=torch.bool, device=sole_points.device)
    elif valid.shape != (batch_size, frames):
        raise ValueError(f"valid must have shape {(batch_size, frames)}, got {tuple(valid.shape)}")
    else:
        valid = valid.to(device=sole_points.device, dtype=torch.bool)

    radii = torch.as_tensor(sole_radii, device=sole_points.device, dtype=sole_points.dtype).reshape(-1)
    if radii.numel() != sole_points.shape[2]:
        raise ValueError("sole_radii must contain one radius per sole proxy point")
    fps_vector = torch.as_tensor(fps, device=sole_points.device, dtype=sole_points.dtype).reshape(-1)
    if fps_vector.numel() == 1:
        fps_vector = fps_vector.expand(batch_size)
    if fps_vector.numel() != batch_size or torch.any(fps_vector <= 0):
        raise ValueError("fps must be positive and scalar or have one value per batch item")

    sole_bottom = sole_points[..., 2] - radii.view(1, 1, -1)
    point_contact = (sole_bottom >= -float(contact_penetration_tolerance)) & (
        sole_bottom <= float(contact_height_threshold)
    )
    point_speed = torch.diff(sole_points[..., :2], dim=1).norm(dim=-1) * fps_vector.view(-1, 1, 1)

    if sole_foot_ids is None:
        foot_groups = (torch.ones(sole_points.shape[2], dtype=torch.bool, device=sole_points.device),)
    else:
        foot_ids = torch.as_tensor(sole_foot_ids, device=sole_points.device).reshape(-1)
        if foot_ids.numel() != sole_points.shape[2]:
            raise ValueError("sole_foot_ids must contain one foot id per sole proxy point")
        foot_groups = tuple(foot_ids == foot_id for foot_id in torch.unique(foot_ids, sorted=True))

    speeds, masks = [], []
    valid_pair = valid[:, 1:] & valid[:, :-1]
    for proxy_mask in foot_groups:
        foot_contact = point_contact[..., proxy_mask].any(dim=-1)
        masks.append(foot_contact[:, 1:] & foot_contact[:, :-1] & valid_pair)
        speeds.append(point_speed[..., proxy_mask].max(dim=-1).values)
    interval_speed = torch.stack(speeds, dim=-1)
    contact_interval = torch.stack(masks, dim=-1)
    numerator = (interval_speed * contact_interval.to(interval_speed.dtype)).sum(dim=(1, 2))
    contact_count = contact_interval.sum(dim=(1, 2))
    valid_count = valid_pair.sum(dim=1) * len(foot_groups)
    value = numerator / contact_count.to(interval_speed.dtype).clamp_min(1.0)
    return value, contact_count, valid_count


class ContactSlidingEvaluator:
    """G1 FK adapter with length-aware GPU batch support."""

    def __init__(self, *, device: str = "auto", contact_height_threshold: float = 0.12,
                 contact_penetration_tolerance: float = 0.02) -> None:
        import torch

        if not G1_KINEMATICS_PATH.is_file():
            raise FileNotFoundError(f"G1 kinematics asset is missing: {G1_KINEMATICS_PATH}")
        if str(OMG_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(OMG_SOURCE_ROOT))
        from omg.robots.g1.kinematics import G1Kinematics

        self.torch = torch
        self.device = _resolve_device(device, torch)
        self.contact_height_threshold = float(contact_height_threshold)
        self.contact_penetration_tolerance = float(contact_penetration_tolerance)
        self.kinematics = G1Kinematics(kinematics_path=str(G1_KINEMATICS_PATH)).to(self.device).eval()

    def score_batch(self, qpos_36: np.ndarray, *, valid: np.ndarray, fps: float | np.ndarray) -> list[ContactSlidingScore]:
        """Score padded complete motions in one FK invocation on the selected GPU."""
        torch = self.torch
        qpos = np.asarray(qpos_36, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if qpos.ndim != 3 or qpos.shape[-1] != 36 or qpos.shape[0] == 0:
            raise ValueError(f"qpos_36 must have shape (B, T, 36), got {qpos.shape}")
        if mask.shape != qpos.shape[:2] or np.any(mask.sum(axis=1) < 2):
            raise ValueError("valid must match qpos and contain at least two real frames per motion")
        with torch.inference_mode():
            qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            valid_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            fk = self.kinematics.forward_kinematics(qpos_tensor)
            sole_points, sole_radii = self.kinematics.get_sole_proxy_points(
                fk["body_pos_w"], fk["body_quat_w"]
            )
            values, contact_counts, valid_counts = contact_sliding_from_sole_proxies(
                sole_points=sole_points, sole_radii=sole_radii, fps=fps,
                contact_height_threshold=self.contact_height_threshold,
                contact_penetration_tolerance=self.contact_penetration_tolerance,
                valid=valid_tensor, sole_foot_ids=self.kinematics.sole_proxy_foot_ids,
            )
        return [
            ContactSlidingScore(float(values[row].detach().cpu()), int(contact_counts[row].detach().cpu()),
                                int(valid_counts[row].detach().cpu()))
            for row in range(qpos.shape[0])
        ]

    def score(self, qpos_36: np.ndarray, *, fps: float) -> ContactSlidingScore:
        """Compatibility wrapper for one unpadded complete motion."""
        qpos = np.asarray(qpos_36, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[-1] != 36:
            raise ValueError(f"qpos_36 must have shape (T, 36), got {qpos.shape}")
        return self.score_batch(qpos[None], valid=np.ones((1, len(qpos)), dtype=bool), fps=fps)[0]


def _resolve_device(requested: str, torch_module: Any):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)
