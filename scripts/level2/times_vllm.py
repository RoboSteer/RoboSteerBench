"""Frozen, resumable VideoLLM protocol for Level-2 Times."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import re
import threading
import time
from typing import Any, Iterable

from scripts.level2.order_vllm import (
    _atomic_json,
    _cache_key,
    _post_json,
    _stat_signature,
    index_predictions,
    index_videos,
    sha256_file,
)


PROMPT_VERSION = "times_closed_set_v1"
PARSER_VERSION = "times_strict_json_v1"
CACHE_SCHEMA = "robosteer.times_vllm_cache.v1"
TASK_ID_PREFIX = "L2_times_text_"
TASK_ID_PREFIXES = {"text": TASK_ID_PREFIX, "audio": "L2_times_audio_", "video": "L2_times_video_"}

TIMES_PROMPT = """You are evaluating a rendered human/humanoid motion video.

Target action: {action}

Based only on visible motion, count the number of complete instances of the
target action.

Rules:
1. The requested count is not provided. Do not guess common benchmark values.
2. Count only clearly and completely executed target actions.
3. A continuously sustained action counts once.
4. A new instance requires the previous instance to end or reset.
5. For cyclic actions, count complete cycles, not frames or intermediate poses.
6. An incomplete instance at the beginning or end does not count.
7. Repetitions of unrelated motion do not count.
8. Use "5+" when more than five complete instances are visible.
9. Return only valid JSON.

Output:
{{"visible": true, "count": 0 | 1 | 2 | 3 | 4 | 5 | "5+"}}"""

TIMES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "count": {
            "anyOf": [
                {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]},
                {"type": "string", "enum": ["5+"]},
            ]
        },
    },
    "required": ["visible", "count"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TimesTaskRecord:
    sample_id: str
    source_sample_id: str
    task_id: str
    task_json: Path
    duration_seconds: float
    condition_text: str
    motion_groundtruth: str
    action: str
    target_count: int
    condition_audio: str | None = None
    condition_videos: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimesTask(TimesTaskRecord):
    video_path: Path | None = None
    prediction_path: Path | None = None


def _validate_gt_annotation(path: Path, action: str, target_count: int) -> None:
    """Cross-check the frozen action/count against optional GT provenance."""
    if not path.is_file():
        raise FileNotFoundError(f"Times motion GT does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: Times motion GT must be a dictionary")
    if payload.get("repeat_count") is not None and int(payload["repeat_count"]) != target_count:
        raise ValueError(f"{path}: GT repeat_count disagrees with task count {target_count}")
    if payload.get("source_proc_label") is not None:
        source_action = payload["source_proc_label"]
        if not isinstance(source_action, str) or source_action.strip() != action:
            raise ValueError(f"{path}: GT source_proc_label disagrees with action {action!r}")


def load_times_records(
    task_root: Path, dataset_root: Path, modality: str = "text",
    action_records: Iterable[TimesTaskRecord] | None = None,
) -> list[TimesTaskRecord]:
    try:
        task_id_prefix = TASK_ID_PREFIXES[modality]
    except KeyError as exc:
        raise ValueError(f"unsupported Times modality: {modality}") from exc
    action_by_sample = (
        {record.sample_id: record for record in action_records}
        if action_records is not None else None
    )
    if modality in {"audio", "video"} and action_by_sample is None:
        raise ValueError(f"{modality} Times requires paired Text task records for action labels")
    task_root = task_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not task_root.is_dir():
        raise NotADirectoryError(f"Times task root does not exist: {task_root}")
    records: list[TimesTaskRecord] = []
    seen: set[str] = set()
    for path in sorted(task_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            task_id = metadata["task_id"]
            family = metadata["task_family"]
            task_type = metadata["task_type"]
            duration = float(metadata["duration"])
            modalities = payload["input"]["modalities"]
            groundtruth = payload["ground_truth"]["motion_parameters"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Times task JSON {path}: {exc}") from exc
        if not isinstance(task_id, str) or not task_id.startswith(task_id_prefix):
            raise ValueError(f"{path}: unexpected Times task_id {task_id!r}")
        sample_id = task_id[len(task_id_prefix):]
        match = re.fullmatch(r"(.+)_([234])x", sample_id)
        if match is None:
            raise ValueError(f"{path}: Times task_id must end in _2x, _3x, or _4x")
        if str(family).casefold() != "times":
            raise ValueError(f"{path}: expected metadata.task_family == 'Times'")
        target_count = int(match.group(2))
        if task_type != f"{target_count}x":
            raise ValueError(f"{path}: metadata.task_type disagrees with task_id count")
        condition_audio = None
        if modality in {"audio", "video"}:
            if modality == "video":
                videos = modalities.get("videos_processed") if isinstance(modalities, dict) else None
                if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str) or not videos[0]:
                    raise ValueError(f"{path}: input.modalities.videos_processed must contain one path")
            audios = modalities.get("audios") if isinstance(modalities, dict) else None
            if modality == "audio":
                if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str) or not audios[0]:
                    raise ValueError(f"{path}: input.modalities.audios must contain one path")
                condition_audio = audios[0]
            label = action_by_sample.get(sample_id)
            if label is None:
                raise ValueError(f"{path}: missing paired Text Times action label")
            if (label.motion_groundtruth != groundtruth or label.duration_seconds != duration
                    or label.target_count != target_count):
                raise ValueError(f"{path}: Text/Audio Times task mismatch")
            condition = label.condition_text
        else:
            condition = modalities.get("text") if isinstance(modalities, dict) else None
        if not isinstance(condition, str):
            raise ValueError(f"{path}: input.modalities.text must be a string")
        prompt = re.fullmatch(r"repeat (.+) ([234]) times", condition.strip())
        if prompt is None or int(prompt.group(2)) != target_count:
            raise ValueError(f"{path}: text condition disagrees with Times task count")
        action = prompt.group(1).strip()
        if not action:
            raise ValueError(f"{path}: target action is empty")
        if not isinstance(groundtruth, str) or not groundtruth:
            raise ValueError(f"{path}: ground_truth.motion_parameters must be a path")
        original_motion = metadata.get("original_motion")
        if original_motion is not None and original_motion != groundtruth:
            raise ValueError(f"{path}: metadata.original_motion disagrees with motion GT")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{path}: metadata.duration must be finite and positive")
        if sample_id in seen:
            raise ValueError(f"duplicate Times task sample ID {sample_id}")
        seen.add(sample_id)
        gt_path = Path(groundtruth)
        gt_path = gt_path.resolve() if gt_path.is_absolute() else (dataset_root / gt_path).resolve()
        _validate_gt_annotation(gt_path, action, target_count)
        records.append(TimesTaskRecord(
            sample_id=sample_id, source_sample_id=match.group(1), task_id=task_id,
            task_json=path, duration_seconds=duration, condition_text=condition.strip(),
            condition_audio=condition_audio,
            condition_videos=tuple(modalities.get("videos_processed", ())) if modality == "video" else (),
            motion_groundtruth=groundtruth, action=action, target_count=target_count,
        ))
    if not records:
        raise RuntimeError(f"no Times task JSON files found under {task_root}")
    return sorted(records, key=lambda record: record.sample_id)


def build_times_tasks(
    records: Iterable[TimesTaskRecord], video_root: Path, prediction_root: Path
) -> list[TimesTask]:
    videos = index_videos(video_root)
    predictions = index_predictions(prediction_root)
    return [TimesTask(
        **record.__dict__,
        video_path=videos.get(record.sample_id),
        prediction_path=predictions.get(record.sample_id),
    ) for record in records]


def parse_times_output(raw_output: str) -> dict[str, Any]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = item
        return result

    try:
        value = json.loads(raw_output, object_pairs_hook=unique_fields)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"response is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    if set(value) != {"visible", "count"}:
        raise ValueError("response fields must be exactly ['count', 'visible']")
    if type(value["visible"]) is not bool:
        raise ValueError("visible must be a boolean")
    count = value["count"]
    if not (type(count) is int and 0 <= count <= 5 or count == "5+"):
        raise ValueError("count must be an integer from 0 to 5 or '5+'")
    if value["visible"] is False and count != 0:
        raise ValueError("visible=false requires count=0")
    return value


def score_times(parsed: dict[str, Any], target_count: int) -> int:
    return int(parsed["visible"] is True and parsed["count"] == target_count)


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / "records" / f"{sample_id}.json"


def _cache_key_payload(
    task: TimesTask, *, video_sha256: str, model_revision: str,
    model_manifest_sha256_value: str, fps: float, max_frames: int,
    temperature: float, seed: int, max_tokens: int,
) -> dict[str, Any]:
    return {
        "video_sha256": video_sha256,
        "task_family": "Times",
        "action": task.action,
        "target_count": task.target_count,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "model_revision": model_revision,
        "model_manifest_sha256": model_manifest_sha256_value,
        "sampling_config": {"fps": fps, "max_frames": max_frames},
        "processor_config": {"cap_pixels_per_frame": True},
        "decoding_config": {
            "temperature": temperature, "do_sample": False, "seed": seed,
            "max_tokens": max_tokens,
        },
    }


def load_cached_result(
    task: TimesTask, cache_dir: Path, *, model_revision: str,
    model_manifest_sha256_value: str, fps: float, max_frames: int,
    temperature: float, seed: int, max_tokens: int,
) -> dict[str, Any] | None:
    if task.video_path is None:
        return None
    path = cache_path(cache_dir, task.sample_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CACHE_SCHEMA:
            return None
        if payload.get("video_stat") != _stat_signature(task.video_path):
            return None
        key = _cache_key(_cache_key_payload(
            task, video_sha256=str(payload["video_sha256"]),
            model_revision=model_revision,
            model_manifest_sha256_value=model_manifest_sha256_value,
            fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
            max_tokens=max_tokens,
        ))
        return payload if payload.get("cache_key") == key else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def invalid_asset_result(task: TimesTask, reason: str) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA, "sample_id": task.sample_id,
        "source_sample_id": task.source_sample_id, "task_family": "Times",
        "action": task.action, "target_count": task.target_count,
        "valid": False, "decoded_video": False, "error": reason, "score": 0,
    }


def _video_metadata(path: Path) -> dict[str, Any]:
    """Validate metadata and decode a frame before asking the model."""
    try:
        import av
    except ModuleNotFoundError:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("video cannot be opened")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0.0
            if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
                raise ValueError("invalid width, height, FPS, or duration")
            if not capture.read()[0]:
                raise ValueError("no decodable video frame")
            return {
                "width": width, "height": height, "fps": fps,
                "duration_seconds": duration,
                "container_frame_count": frames,
                "estimated_frame_count": max(1, round(duration * fps)),
                "first_frame_decodable": True,
            }
        finally:
            capture.release()

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if len(streams) != 1:
            raise ValueError(f"expected one video stream, found {len(streams)}")
        stream = streams[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        duration = (
            float(container.duration) / float(av.time_base)
            if container.duration is not None else
            float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        )
        if stream.width <= 0 or stream.height <= 0 or fps <= 0 or duration <= 0:
            raise ValueError("invalid width, height, FPS, or duration")
        if next(container.decode(video=stream.index), None) is None:
            raise ValueError("no decodable video frame")
        return {
            "width": stream.width, "height": stream.height, "fps": fps,
            "duration_seconds": duration,
            "container_frame_count": stream.frames or None,
            "estimated_frame_count": max(1, round(duration * fps)),
            "first_frame_decodable": True,
        }


def infer_one(
    task: TimesTask, endpoint: str, *, cache_dir: Path, served_model_name: str,
    model_revision: str, model_manifest_sha256_value: str, fps: float,
    max_frames: int, temperature: float, seed: int, max_tokens: int,
    request_timeout: float, retries: int,
) -> dict[str, Any]:
    if task.video_path is None:
        return invalid_asset_result(task, "missing rendered prediction video")
    try:
        video_stat = _stat_signature(task.video_path)
        video_hash = sha256_file(task.video_path)
    except OSError as exc:
        return invalid_asset_result(task, f"missing or corrupt rendered prediction video: {exc}")
    key = _cache_key(_cache_key_payload(
        task, video_sha256=video_hash, model_revision=model_revision,
        model_manifest_sha256_value=model_manifest_sha256_value,
        fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
        max_tokens=max_tokens,
    ))
    try:
        video_metadata = _video_metadata(task.video_path)
    except (OSError, ValueError) as exc:
        result = invalid_asset_result(task, f"missing or corrupt rendered prediction video: {exc}")
        result.update({
            "video_path": str(task.video_path), "video_sha256": video_hash,
            "video_stat": video_stat, "cache_key": key,
            "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
            "model_revision": model_revision,
            "model_manifest_sha256": model_manifest_sha256_value,
            "sampling_config": {"fps": fps, "max_frames": max_frames},
            "processor_config": {"cap_pixels_per_frame": True},
            "decoding_config": {
                "temperature": temperature, "do_sample": False, "seed": seed,
                "max_tokens": max_tokens,
            },
        })
        _atomic_json(cache_path(cache_dir, task.sample_id), result)
        return result
    request_payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": task.video_path.as_uri()}},
            {"type": "text", "text": TIMES_PROMPT.format(action=task.action)},
        ]}],
        "temperature": temperature, "seed": seed, "max_tokens": max_tokens,
        "media_io_kwargs": {"video": {"fps": fps, "num_frames": max_frames}},
        "mm_processor_kwargs": {"cap_pixels_per_frame": True},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "times_evaluation", "strict": True, "schema": TIMES_JSON_SCHEMA,
        }},
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            status, response_body = _post_json(endpoint, request_payload, request_timeout)
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {response_body[:2000]}")
            response = json.loads(response_body)
            raw_output = response["choices"][0]["message"]["content"]
            try:
                parsed_output = parse_times_output(raw_output)
                valid, error = True, None
                score = score_times(parsed_output, task.target_count)
            except ValueError as exc:
                parsed_output, valid, error, score = None, False, str(exc), 0
            result = {
                "schema": CACHE_SCHEMA, "sample_id": task.sample_id,
                "source_sample_id": task.source_sample_id,
                "video_path": str(task.video_path), "video_sha256": video_hash,
                "video_stat": video_stat, "video_metadata": video_metadata,
                "task_family": "Times", "action": task.action,
                "target_count": task.target_count,
                "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
                "model_revision": model_revision,
                "model_manifest_sha256": model_manifest_sha256_value,
                "sampling_config": {"fps": fps, "max_frames": max_frames},
                "processor_config": {"cap_pixels_per_frame": True},
                "decoding_config": {
                    "temperature": temperature, "do_sample": False, "seed": seed,
                    "max_tokens": max_tokens,
                },
                "endpoint": endpoint, "raw_output": raw_output,
                "parsed_output": parsed_output, "valid": valid,
                "decoded_video": True, "error": error, "score": score,
                "usage": response.get("usage"), "cache_key": key,
            }
            _atomic_json(cache_path(cache_dir, task.sample_id), result)
            return result
        except (OSError, TimeoutError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"infrastructure request failure for {task.sample_id} via {endpoint}: {last_error}"
    )


def run_inference(
    tasks: list[TimesTask], endpoints: list[str], *, cache_dir: Path,
    served_model_name: str, model_revision: str, model_manifest_sha256_value: str,
    fps: float, max_frames: int, temperature: float, seed: int,
    max_tokens: int, request_timeout: float, retries: int,
    requests_per_server: int, progress_path: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pending: list[TimesTask] = []
    for task in tasks:
        if task.video_path is None:
            results[task.sample_id] = invalid_asset_result(task, "missing rendered prediction video")
            continue
        cached = load_cached_result(
            task, cache_dir, model_revision=model_revision,
            model_manifest_sha256_value=model_manifest_sha256_value,
            fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
            max_tokens=max_tokens,
        )
        if cached is None:
            pending.append(task)
        else:
            results[task.sample_id] = cached
    if pending and not endpoints:
        raise RuntimeError(f"{len(pending)} uncached Times tasks remain without a vLLM endpoint")
    total = len(tasks)
    started = time.monotonic()
    _atomic_json(progress_path, {
        "expected": total, "cached": len(results), "pending": len(pending),
        "completed": len(results), "status": "running", "updated_at": time.time(),
    })
    if not pending:
        _atomic_json(progress_path, {
            "expected": total, "completed": len(results), "status": "complete",
            "elapsed_seconds": time.monotonic() - started, "updated_at": time.time(),
        })
        return results

    semaphores = {endpoint: threading.Semaphore(requests_per_server) for endpoint in endpoints}

    def execute(task: TimesTask, endpoint: str) -> dict[str, Any]:
        with semaphores[endpoint]:
            return infer_one(
                task, endpoint, cache_dir=cache_dir, served_model_name=served_model_name,
                model_revision=model_revision,
                model_manifest_sha256_value=model_manifest_sha256_value,
                fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
                max_tokens=max_tokens, request_timeout=request_timeout, retries=retries,
            )

    workers = len(endpoints) * requests_per_server
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="times-vllm") as executor:
        future_to_task = {
            executor.submit(execute, task, endpoints[index % len(endpoints)]): task
            for index, task in enumerate(pending)
        }
        completed_new = 0
        last_report = time.monotonic()
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            results[task.sample_id] = future.result()
            completed_new += 1
            now = time.monotonic()
            if completed_new % 25 == 0 or now - last_report >= 30 or completed_new == len(pending):
                elapsed = max(now - started, 1e-6)
                rate = completed_new / elapsed
                eta = (len(pending) - completed_new) / rate if rate > 0 else None
                _atomic_json(progress_path, {
                    "expected": total, "cached_at_start": total - len(pending),
                    "pending_at_start": len(pending), "completed": len(results),
                    "completed_new": completed_new, "rate_samples_per_second": rate,
                    "eta_seconds": eta, "status": "running", "updated_at": time.time(),
                })
                print(
                    f"Times VideoLLM progress: {len(results)}/{total} "
                    f"({rate:.2f} new samples/s, ETA {eta / 60:.1f} min)" if eta is not None
                    else f"Times VideoLLM progress: {len(results)}/{total}",
                    flush=True,
                )
                last_report = now
    _atomic_json(progress_path, {
        "expected": total, "completed": len(results), "status": "complete",
        "elapsed_seconds": time.monotonic() - started, "updated_at": time.time(),
    })
    return results
