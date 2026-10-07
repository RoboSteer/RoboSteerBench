from __future__ import annotations

import os
import re
import time
import hashlib
import subprocess
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, Any]:
    """Read lightweight FFmpeg header metadata without starting a frame pipe.

    The former implementation started ``read_frames()``, consumed only its
    header, and then closed the raw-video pipe.  FFmpeg can remain blocked on
    that pipe and imageio-ffmpeg waits 1.5 seconds before killing it.  Across
    tens of thousands of videos this is both noisy and very expensive.
    """
    video_path = Path(path).resolve()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(video_path)]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"FFmpeg metadata probe timed out for {video_path}") from exc
    header = completed.stderr
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", header)
    video_match = re.search(
        r"Video:.*?,\s*(\d+)x(\d+).*?,\s*(\d+(?:\.\d+)?)\s*fps(?:,|\s)",
        header,
    )
    if duration_match is None or video_match is None:
        raise ValueError(f"Invalid FFmpeg video metadata for {video_path}: {header[-1000:]}")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    width, height, fps_text = video_match.groups()
    fps = float(fps_text)
    frames = int(round(fps * duration))
    if fps <= 0 or frames <= 0:
        raise ValueError(
            f"Invalid FFmpeg video metadata for {video_path}: fps={fps}, frames={frames}"
        )
    stat = video_path.stat()
    return {
        "source": str(video_path),
        # This identity avoids a full second read of every source video while
        # retaining enough provenance to diagnose stale or replaced inputs.
        "source_file_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_fps": fps,
        "source_frames": frames,
        "duration_seconds": duration,
        "width": int(width),
        "height": int(height),
        "decoder": "imageio_ffmpeg",
        "decoder_version": imageio_ffmpeg.__version__,
    }


def sample_video_for_motion_windows(
    path: Path,
    motion_window_records: list[Any],
    *,
    derived_fps: float,
    frames_per_window: int,
    inspected_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = dict(inspected_metadata) if inspected_metadata is not None else inspect_video(path)
    video_fps = float(metadata["source_fps"])
    video_frames = int(metadata["source_frames"])
    requested_times: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    for record in motion_window_records:
        start_time = float(record.start_frame_30fps / derived_fps)
        end_frame = max(record.start_frame_30fps, record.end_frame_exclusive_30fps - 1)
        end_time = float(end_frame / derived_fps)
        times = np.linspace(start_time, end_time, num=frames_per_window, dtype=np.float64)
        indices = np.rint(times * video_fps).astype(np.int64)
        indices = np.clip(indices, 0, video_frames - 1)
        requested_times.append(times)
        frame_indices.append(indices)
    all_indices = sorted(set(np.concatenate(frame_indices).tolist()))
    decoded: dict[int, np.ndarray] = {}
    # Ask FFmpeg to transport only the exact source-frame indices used by the
    # evaluator. The decoder still observes the same compressed stream and
    # rgb24 conversion, but irrelevant raw frames no longer cross the pipe.
    select_filter = "select=" + "+".join(f"eq(n\\,{index})" for index in all_indices)
    reader = imageio_ffmpeg.read_frames(
        str(path),
        pix_fmt="rgb24",
        output_params=[
            "-vf",
            select_filter,
            "-vsync",
            "0",
            "-frames:v",
            str(len(all_indices)),
        ],
    )
    try:
        frame_metadata = next(reader)
        frame_width, frame_height = (int(value) for value in frame_metadata["size"])
        if (frame_width, frame_height) != (int(metadata["width"]), int(metadata["height"])):
            raise ValueError(f"FFmpeg frame size changed while decoding {path}")
        for source_index in all_indices:
            try:
                frame_bytes = next(reader)
            except StopIteration:
                break
            decoded[source_index] = np.frombuffer(
                frame_bytes, dtype=np.uint8
            ).reshape(frame_height, frame_width, 3).copy()
        # One final pull lets imageio-ffmpeg observe the bounded subprocess's
        # natural EOF. Closing an active raw-video pipe is what caused the
        # repeated 1.5-second wait and "We had to kill ffmpeg" message.
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise ValueError(f"FFmpeg emitted too many selected frames for {path}")
    finally:
        reader.close()
    missing = [index for index in all_indices if index not in decoded]
    if missing:
        raise ValueError(f"Video decode ended early for {path}; missing frame indices {missing[:8]}")
    windows = np.stack([np.stack([decoded[int(index)] for index in indices]) for indices in frame_indices])
    metadata.update(
        {
            "sampling_policy": "motion-window timestamps, nearest video frame",
            "frames_per_window": frames_per_window,
            "motion_timebase_fps": float(derived_fps),
            "windows": [
                {
                    "index": index,
                    "requested_times_seconds": requested_times[index].tolist(),
                    "sampled_frame_indices": frame_indices[index].tolist(),
                    "sampled_times_seconds": (frame_indices[index] / video_fps).tolist(),
                }
                for index in range(len(frame_indices))
            ],
        }
    )
    return windows, metadata


class FrozenXCLIPVideoEncoder:
    def __init__(self, config: Any, device: torch.device):
        # Standard HTTP has been more stable than Xet for this Windows environment.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from transformers import AutoModel, AutoProcessor

        config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[video encoder] loading {config.xclip_model}@{config.xclip_revision}", flush=True)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            config.xclip_model,
            revision=config.xclip_revision,
            cache_dir=config.model_cache_dir,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            config.xclip_model,
            revision=config.xclip_revision,
            cache_dir=config.model_cache_dir,
            local_files_only=True,
        ).to(device)
        self.model.requires_grad_(False).eval()
        self.device = device
        self.model_name = config.xclip_model
        self.revision = config.xclip_revision
        self.output_dim = int(self.model.config.projection_dim)
        self.last_effective_batch_size = 0
        print(f"[video encoder] ready in {time.perf_counter() - started:.1f}s", flush=True)

    @torch.inference_mode()
    def encode_window_features(
        self,
        windows_rgb: Sequence[np.ndarray] | np.ndarray,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Encode arbitrary video windows, allowing mixed source resolutions.

        The batch size automatically backs off on CUDA OOM. No mixed precision is
        used so features remain compatible with checkpoints created by the first
        extraction implementation.
        """
        if len(windows_rgb) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)
        features: list[torch.Tensor] = []
        start = 0
        effective_batch_size = max(1, int(batch_size))
        while start < len(windows_rgb):
            current_size = min(effective_batch_size, len(windows_rgb) - start)
            clips = [windows_rgb[index] for index in range(start, start + current_size)]
            try:
                inputs = self.processor(videos=[list(clip) for clip in clips], return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                if self.device.type == "cuda":
                    pixel_values = pixel_values.pin_memory().to(self.device, non_blocking=True)
                else:
                    pixel_values = pixel_values.to(self.device)
                outputs = self.model.get_video_features(pixel_values=pixel_values)
                pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                features.append(F.normalize(pooled, dim=-1).cpu())
                start += current_size
                self.last_effective_batch_size = max(self.last_effective_batch_size, current_size)
                del inputs, pixel_values, outputs, pooled
            except torch.cuda.OutOfMemoryError:
                if current_size == 1:
                    raise
                effective_batch_size = max(1, current_size // 2)
                self.last_effective_batch_size = 0
                torch.cuda.empty_cache()
                print(
                    f"[video encoder] CUDA OOM; retrying with window batch={effective_batch_size}",
                    flush=True,
                )
        return torch.cat(features, dim=0).numpy().astype(np.float32)

    def encode_windows(self, windows_rgb: np.ndarray, batch_size: int = 8) -> np.ndarray:
        window_features = torch.from_numpy(self.encode_window_features(windows_rgb, batch_size=batch_size))
        return F.normalize(window_features.mean(dim=0, keepdim=True), dim=-1)[0].numpy().astype(np.float32)
