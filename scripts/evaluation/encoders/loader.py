"""Load registered encoders from the selected model registry."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm


@dataclass
class RegisteredEncoder:
    name: str
    model: Any
    device: Any
    input_key: str
    flatten_features: bool
    l2_normalize: bool
    batch_size: int

    def encode(self, values: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Encode all inputs in configured-size batches."""
        array = _stack(values)
        outputs: list[np.ndarray] = []
        total_batches = (len(array) + self.batch_size - 1) // self.batch_size
        for start in tqdm(
            range(0, len(array), self.batch_size),
            total=total_batches,
            desc=f"Encoding {self.name}",
            unit="batch",
            dynamic_ncols=True,
        ):
            outputs.append(self.encode_batch(array[start : start + self.batch_size]))
        result = np.concatenate(outputs, axis=0).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"encoder {self.name!r} produced non-finite embeddings")
        return result

    def encode_batch(self, values: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Encode one bounded batch without creating a nested progress bar."""
        import torch
        import torch.nn.functional as functional

        array = _stack(values)
        if len(array) > self.batch_size:
            raise ValueError(
                f"batch has {len(array)} items, exceeding configured batch_size={self.batch_size}"
            )
        if self.flatten_features and array.ndim >= 4:
            array = array.reshape(*array.shape[:2], -1)
        with torch.inference_mode():
            batch = torch.as_tensor(array, dtype=torch.float32, device=self.device)
            embedding = _extract_embedding(self.model(batch))
            if embedding.ndim > 2:
                embedding = embedding.mean(dim=1)
            if embedding.ndim != 2:
                raise ValueError(
                    f"encoder {self.name!r} must produce (B,D) embeddings; got {tuple(embedding.shape)}"
                )
            if self.l2_normalize:
                embedding = functional.normalize(embedding, dim=-1)
            result = embedding.detach().cpu().float().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"encoder {self.name!r} produced non-finite embeddings")
        return result.astype(np.float32, copy=False)


def load_registry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"encoder registry does not exist: {path}. Provide the downloaded models/ bundle "
            "or pass --models explicitly."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"encoder registry must be a JSON object: {path}")
    return payload


def load_encoder(name: str, registry_path: Path, device: str) -> RegisteredEncoder:
    import torch

    registry = load_registry(registry_path)
    if name not in registry:
        raise KeyError(f"encoder {name!r} is not registered in {registry_path}")
    config = registry[name]
    encoder_type = str(config.get("type", "torch"))
    if encoder_type not in {"torch", "torchscript", "omg_motion"}:
        raise ValueError(
            f"unsupported encoder type {encoder_type!r} for {name!r}; "
            "currently supported: torch, torchscript, omg_motion"
        )
    checkpoint = _resolve_path(Path(config["checkpoint"]), registry_path.resolve().parent)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint for encoder {name!r} does not exist: {checkpoint}")
    torch_device = _device(device, torch)
    if encoder_type == "omg_motion":
        model = _load_omg_motion_encoder(checkpoint, config, registry_path.resolve().parent, torch_device, torch)
    else:
        model = _load_torch_model(checkpoint, torch_device, encoder_type, torch)
    return RegisteredEncoder(
        name=name,
        model=model,
        device=torch_device,
        input_key=str(config.get("input_key", "features")),
        flatten_features=bool(config.get("flatten_features", True)),
        l2_normalize=bool(config.get("l2_normalize", False)),
        batch_size=int(config.get("batch_size", 32)),
    )


def _load_omg_motion_encoder(checkpoint: Path, config: dict[str, Any], registry_root: Path, device: Any, torch: Any) -> Any:
    architecture = _resolve_path(Path(config.get("architecture", checkpoint.parent / "model.py")), registry_root)
    if not architecture.is_file():
        raise FileNotFoundError(f"OMG MotionEncoder architecture does not exist: {architecture}")
    module_spec = importlib.util.spec_from_file_location("liujiahui_omg_motion_encoder", architecture)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"unable to import MotionEncoder architecture from {architecture}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "motion_encoder" not in payload or "config" not in payload:
        raise ValueError(f"invalid OMG MotionEncoder checkpoint: {checkpoint}")
    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"invalid OMG MotionEncoder config in {checkpoint}")
    model = module.MotionEncoder(
        input_dim=int(checkpoint_config["input_dim"]), movement_dim=int(checkpoint_config["movement_dim"]),
        hidden_dim=int(checkpoint_config["hidden_dim"]), output_dim=int(checkpoint_config["embedding_dim"]),
        movement_mode=str(checkpoint_config["movement_mode"]), temporal_kind=str(checkpoint_config["temporal_kind"]),
        num_layers=int(checkpoint_config["num_layers"]), num_heads=int(checkpoint_config["num_heads"]),
        mlp_ratio=float(checkpoint_config["mlp_ratio"]), dropout=float(checkpoint_config["dropout"]),
        max_len=int(checkpoint_config["max_len"]), normalize=bool(checkpoint_config["normalize"]),
    )
    model.load_state_dict(payload["motion_encoder"], strict=True)
    return model.to(device).eval()


def _load_torch_model(checkpoint: Path, device: Any, model_type: str, torch: Any) -> Any:
    if model_type == "torchscript":
        model = torch.jit.load(str(checkpoint), map_location=device)
    else:
        try:
            model = torch.jit.load(str(checkpoint), map_location=device)
        except Exception:
            model = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if not callable(model) or not hasattr(model, "eval"):
        raise TypeError(
            f"{checkpoint} is not a callable serialized module. A state_dict requires an "
            "encoder-specific factory and cannot be loaded generically."
        )
    return model.to(device).eval()


def _extract_embedding(output: Any):
    if hasattr(output, "ndim"):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _extract_embedding(output[0])
    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "motion_embedding", "latent", "output"):
            if key in output:
                return _extract_embedding(output[key])
    raise TypeError(f"cannot extract an embedding tensor from encoder output {type(output)!r}")


def _stack(values: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = values
    else:
        try:
            array = np.stack([np.asarray(value, dtype=np.float32) for value in values])
        except ValueError as exc:
            raise ValueError("encoder inputs must have a common shape") from exc
    if array.ndim < 2 or len(array) == 0:
        raise ValueError(f"encoder input must contain a batch dimension, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("encoder input contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _resolve_path(path: Path, base: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _device(requested: str, torch: Any):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)
