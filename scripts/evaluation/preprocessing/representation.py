"""Convert qpos batches to the representation requested by a MotionEncoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scripts.evaluation.omg_paths import omg_root


OMG_ROOT = omg_root()


def motion_representation(qpos_36: np.ndarray, input_key: str, device: str = "cpu") -> np.ndarray:
    if input_key == "qpos_36":
        return np.asarray(qpos_36, dtype=np.float32)
    if input_key not in {"body_pos_local", "body_link_pos_local"}:
        raise ValueError(
            f"unsupported motion encoder input_key={input_key!r}; expected qpos_36, "
            "body_pos_local, or body_link_pos_local"
        )
    source_root = OMG_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import torch
        from omg.benchmarks.evaluator.representation import canonical_body_positions_from_qpos
        from omg.robots.g1.kinematics import G1Kinematics
    except ImportError as exc:
        raise RuntimeError(
            "body_pos_local preprocessing requires OMG; set ROBOOSTEER_OMG_ROOT"
        ) from exc
    torch_device = _torch_device(device, torch)
    kinematics = G1Kinematics(
        kinematics_path=str(OMG_ROOT / "assets/robots/g1/g1_kinematics.json")
    )
    with torch.inference_mode():
        tensor = torch.as_tensor(qpos_36, dtype=torch.float32, device=torch_device)
        local = canonical_body_positions_from_qpos(tensor, kinematics)
        if input_key == "body_link_pos_local":
            local = local[..., 1:, :]
    return local.detach().cpu().numpy().astype(np.float32)


def _torch_device(requested: str, torch_module: object):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)
