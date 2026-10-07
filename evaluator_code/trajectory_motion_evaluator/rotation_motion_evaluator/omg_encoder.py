from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .runtime import sha256_file


def _motion_encoder_kwargs(config: dict[str, Any], input_dim: int, output_dim: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_dim": input_dim, "output_dim": output_dim}
    for key in (
        "movement_dim", "hidden_dim", "movement_mode", "temporal_kind",
        "num_layers", "num_heads", "mlp_ratio", "dropout", "max_len",
    ):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


class FrozenOMGMotionEncoder:
    def __init__(
        self,
        omg_root: Path,
        omg_checkpoint: Path,
        device: torch.device,
        embedding_dim: int = 512,
    ):
        omg_root = Path(omg_root).resolve()
        omg_checkpoint = Path(omg_checkpoint).resolve()
        source = str(omg_root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from omg.benchmarks.evaluator.motion_encoder import MotionEncoder
        from omg.benchmarks.evaluator.representation import motion_input_dim
        from omg.robots.g1.kinematics import G1Kinematics

        # PyTorch 2.2 only accepts a string path when mmap=True is requested.
        checkpoint = torch.load(str(omg_checkpoint), map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(checkpoint, dict) or "motion_encoder" not in checkpoint:
            raise ValueError(f"Invalid OMG evaluator checkpoint: {omg_checkpoint}")
        config = checkpoint.get("config") or {}
        motion_key = str(config.get("motion_key", "body_pos_local"))
        if motion_key != "body_pos_local":
            raise ValueError(f"Expected OMG motion_key='body_pos_local', got {motion_key!r}")
        relative_kinematics = Path(str(config.get("kinematics_path", "assets/robots/g1/g1_kinematics.json")))
        kinematics_path = relative_kinematics if relative_kinematics.is_absolute() else omg_root / relative_kinematics
        input_dim = motion_input_dim(motion_key, kinematics_path=str(kinematics_path))
        output_dim = int(config.get("embedding_dim", embedding_dim))
        self.encoder = MotionEncoder(**_motion_encoder_kwargs(config, input_dim, output_dim)).to(device)
        self.encoder.load_state_dict(checkpoint["motion_encoder"], strict=True)
        self.encoder.requires_grad_(False).eval()
        self.kinematics = G1Kinematics(kinematics_path=str(kinematics_path))
        self.device = device
        self.checkpoint_sha256 = sha256_file(omg_checkpoint)

    @torch.inference_mode()
    def encode_windows(self, qpos_windows: np.ndarray, valid_masks: np.ndarray) -> np.ndarray:
        from omg.benchmarks.evaluator.representation import canonical_body_positions_from_qpos

        qpos = torch.as_tensor(qpos_windows, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(valid_masks, dtype=torch.bool, device=self.device)
        body_pos_local = canonical_body_positions_from_qpos(qpos, self.kinematics)
        window_embeddings = self.encoder(body_pos_local, valid_mask=valid)
        return F.normalize(window_embeddings.mean(dim=0, keepdim=True), dim=-1)[0].cpu().numpy().astype(np.float32)
