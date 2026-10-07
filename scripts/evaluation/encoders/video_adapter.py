"""Adapter for the delivered Human/Skeleton Video--Motion evaluators."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.data.instruction import InstructionIndex, InstructionSample
from scripts.evaluation.encoders.video_cache import LocalVideoCache
from scripts.evaluation.encoders.cross_modal import CrossModalMotionEvaluator, _resolve_path


class VideoMotionAdapter(CrossModalMotionEvaluator):
    """Encode an MP4 condition and a predicted qpos trajectory in a shared space."""

    _VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    _SEGMENT_AGGREGATION = "ordered_segment_embeddings_l2_normalized_mean_v1"

    def __init__(self, name: str, config: dict[str, Any], registry_root: Path, device: str) -> None:
        super().__init__(name, config, registry_root, device)
        delivery_root = _resolve_path(config["root"], registry_root)
        checkpoint = _resolve_path(config["checkpoint"], registry_root)
        omg_root = _resolve_path(config["omg_root"], registry_root)
        omg_checkpoint = _resolve_path(config["omg_checkpoint"], registry_root)
        xclip_cache = _resolve_path(config["xclip_cache"], registry_root)
        xclip_model_path = (
            _resolve_path(config["xclip_model_path"], registry_root)
            if config.get("xclip_model_path") else None
        )
        video_kind = str(config["video_kind"])
        if video_kind not in {"human", "skeleton"}:
            raise ValueError(f"{name} video_kind must be 'human' or 'skeleton', got {video_kind!r}")
        for label, path, is_dir in (
            ("video evaluator root", delivery_root, True),
            ("video evaluator checkpoint", checkpoint, False),
            ("OMG root", omg_root, True),
            ("OMG motion checkpoint", omg_checkpoint, False),
            ("offline XCLIP cache", xclip_cache, True),
            *([("XCLIP backbone", xclip_model_path, True)] if xclip_model_path else []),
        ):
            if not (path.is_dir() if is_dir else path.is_file()):
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if str(delivery_root) not in sys.path:
            sys.path.insert(0, str(delivery_root))
        try:
            from video_motion_evaluator.inference import VideoMotionEvaluator
        except ImportError as exc:
            raise RuntimeError(
                f"{name} requires the delivered video_motion_evaluator package and its dependencies"
            ) from exc

        evaluator_device = None if device == "auto" else device
        self._evaluator = VideoMotionEvaluator(
            checkpoint,
            omg_root=omg_root,
            omg_checkpoint=omg_checkpoint,
            model_cache_dir=xclip_cache,
            xclip_model_path=xclip_model_path,
            expected_video_kind=video_kind,
            device=evaluator_device,
        )
        self._video_kind = video_kind
        self._condition_asset = str(config.get("condition_asset", "video"))
        aggregation = str(config.get("video_condition_aggregation", "")).strip()
        self._condition_aggregation = aggregation or None
        if self._condition_aggregation not in {None, self._SEGMENT_AGGREGATION}:
            raise ValueError(
                f"unsupported video condition aggregation: {self._condition_aggregation!r}"
            )
        # These bound the adapter's internal work even when the shared MM
        # pipeline supplies a much larger pair batch.
        self._video_condition_batch_size = max(1, int(config.get("video_condition_batch_size", 8)))
        self._video_decode_workers = max(1, int(config.get("video_decode_workers", 8)))
        self._xclip_window_batch_size = max(1, int(config.get("xclip_window_batch_size", 32)))
        self._motion_qpos_batch_size = max(1, int(config.get("motion_qpos_batch_size", 256)))
        self._motion_window_batch_size = max(1, int(config.get("motion_window_batch_size", 2048)))
        local_cache_root = str(config.get("video_local_cache_root", "")).strip()
        self._local_video_cache = (
            LocalVideoCache(Path(local_cache_root)) if local_cache_root else None
        )
        self._paths = {
            "checkpoint": str(checkpoint),
            "omg_checkpoint": str(omg_checkpoint),
            "xclip_cache": str(xclip_cache),
        }

    def _source_video_paths(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> list[Path]:
        assets = (getattr(sample, "attributes", {}) or {}).get("assets", {}) or {}
        asset = assets.get(self._condition_asset)
        raw_paths = asset if isinstance(asset, list) else [asset if asset is not None else sample.path]
        if not raw_paths:
            raise ValueError(f"video condition {sample.sample_id!r} has an empty asset list")
        if len(raw_paths) > 1 and self._condition_aggregation != self._SEGMENT_AGGREGATION:
            raise ValueError(
                f"video condition {sample.sample_id!r} has multiple segments but manifest "
                f"aggregation is not {self._SEGMENT_AGGREGATION!r}"
            )
        paths: list[Path] = []
        for raw_path in raw_paths:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = index.root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"video condition does not exist: {path}")
            if path.suffix.lower() not in self._VIDEO_SUFFIXES:
                raise ValueError(
                    f"{self.name} requires a video condition "
                    f"({', '.join(self._VIDEO_SUFFIXES)}), got {path}"
                )
            paths.append(path)
        return paths

    def _decode_video_path(self, source: Path) -> Path:
        if self._local_video_cache is None:
            return source
        return self._local_video_cache.materialize(source)

    def encode_condition(
        self, sample: InstructionSample, index: InstructionIndex
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sources = self._source_video_paths(sample, index)
        paths = [self._decode_video_path(source) for source in sources]
        if len(paths) == 1:
            encoded = [self._evaluator.encode_video(paths[0])]
        else:
            encoded = self._evaluator.encode_video_batch(
                paths,
                decode_workers=self._video_decode_workers,
                xclip_window_batch_size=self._xclip_window_batch_size,
            )
        return self._merge_condition_segments(sample, sources, paths, encoded)

    def _merge_condition_segments(
        self,
        sample: InstructionSample,
        sources: list[Path],
        paths: list[Path],
        encoded: list[tuple[np.ndarray, dict[str, Any]]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if len(encoded) != len(sources) or len(paths) != len(sources):
            raise RuntimeError("video segment encoding count mismatch")
        if len(encoded) == 1:
            embedding, metadata = encoded[0]
            return np.asarray(embedding, dtype=np.float32), {
                "sample_id": sample.sample_id,
                "path": str(sources[0]),
                "decode_path": str(paths[0]),
                **metadata,
            }
        embeddings = np.stack(
            [np.asarray(embedding, dtype=np.float32) for embedding, _ in encoded]
        )
        pooled = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"invalid pooled video embedding for {sample.sample_id!r}")
        pooled = (pooled / norm).astype(np.float32)
        return pooled, {
            "sample_id": sample.sample_id,
            "path": [str(source) for source in sources],
            "decode_path": [str(path) for path in paths],
            "segment_count": len(encoded),
            "segment_aggregation": self._SEGMENT_AGGREGATION,
            "segments": [metadata for _, metadata in encoded],
        }

    def encode_condition_batch(
        self, samples: list[InstructionSample], index: InstructionIndex
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """Decode a small group concurrently, then batch its XCLIP windows."""
        output: list[tuple[np.ndarray, dict[str, Any]]] = []
        for start in range(0, len(samples), self._video_condition_batch_size):
            group = samples[start:start + self._video_condition_batch_size]
            sources_by_sample = [self._source_video_paths(sample, index) for sample in group]
            sources = [source for sample_sources in sources_by_sample for source in sample_sources]
            if self._local_video_cache is None:
                paths = sources
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self._video_decode_workers, len(sources)),
                    thread_name_prefix="video-cache",
                ) as pool:
                    paths = list(pool.map(self._decode_video_path, sources))
            encoded = self._evaluator.encode_video_batch(
                paths,
                decode_workers=self._video_decode_workers,
                xclip_window_batch_size=self._xclip_window_batch_size,
            )
            offset = 0
            for sample, sample_sources in zip(group, sources_by_sample, strict=True):
                count = len(sample_sources)
                output.append(self._merge_condition_segments(
                    sample,
                    sample_sources,
                    paths[offset:offset + count],
                    encoded[offset:offset + count],
                ))
                offset += count
            if offset != len(encoded):
                raise RuntimeError("video segment batch grouping mismatch")
        return output

    def encode_motion_batch(
        self, qpos_36_batch: list[np.ndarray], source_fps: float
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        output: list[tuple[np.ndarray, dict[str, Any]]] = []
        for start in range(0, len(qpos_36_batch), self._motion_qpos_batch_size):
            output.extend(self._evaluator.encode_qpos_batch(
                qpos_36_batch[start:start + self._motion_qpos_batch_size],
                source_fps,
                window_batch_size=self._motion_window_batch_size,
            ))
        return output

    def encode_motion(
        self, qpos_36: np.ndarray, source_fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        embedding, metadata = self._evaluator.encode_qpos(qpos_36, source_fps)
        return np.asarray(embedding, dtype=np.float32), metadata

    def protocol(self) -> dict[str, Any]:
        architecture = self._evaluator.checkpoint_metadata.get("architecture") or {}
        protocol = {
            "evaluator": "VideoMotionEvaluator",
            "condition": f"complete_{self._video_kind}_video",
            "condition_asset": self._condition_asset,
            "motion_input": "complete_qpos_36",
            "motion_target_fps": int(self._evaluator.derived_fps),
            "motion_window_frames": int(self._evaluator.window_frames),
            "motion_window_stride": int(self._evaluator.window_stride),
            "video_frames_per_window": int(self._evaluator.frames_per_window),
            "video_decoder": "imageio_ffmpeg",
            "video_decode_policy": "exact_frame_index_select_rgb24_v1",
            "video_local_cache": self._local_video_cache is not None,
            "video_condition_batch_size": self._video_condition_batch_size,
            "video_decode_workers": self._video_decode_workers,
            "xclip_window_batch_size": self._xclip_window_batch_size,
            "motion_qpos_batch_size": self._motion_qpos_batch_size,
            "motion_window_batch_size": self._motion_window_batch_size,
            "video_backbone": architecture.get("video_backbone"),
            "video_backbone_revision": architecture.get("video_backbone_revision"),
            "embedding_dim": int(architecture.get("embedding_dim", 512)),
            "l2_normalized": True,
            **self._paths,
        }
        if self._condition_aggregation is not None:
            protocol["video_condition_aggregation"] = self._condition_aggregation
        return protocol
