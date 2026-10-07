"""Frozen VideoLLM protocol and resumable runtime for Level-2 Order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
from urllib.parse import urlsplit

PROMPT_VERSION = "order_closed_set_v1"
PARSER_VERSION = "order_strict_json_v1"
CACHE_SCHEMA = "robosteer.order_vllm_cache.v1"
TASK_ID_PREFIX = "L2_order_text_"
TASK_ID_PREFIXES = {"text": TASK_ID_PREFIX, "audio": "L2_order_audio_", "video": "L2_order_video_"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

ORDER_PROMPT = """You are evaluating a rendered human/humanoid motion video.

Candidate action A: {action_a}
Candidate action B: {action_b}

Based only on the visible motion, determine whether each candidate action is
performed. If both are performed, determine which action begins first.

Rules:
1. Do not infer an intended order from the candidate names or display order.
2. Mark an action visible only when its characteristic motion is clearly
   executed. Preparation or an incomplete attempt is insufficient.
3. Determine order from action onset, not peak-motion or completion time.
4. Ignore incidental standing, balancing, locomotion, and transitions.
5. Return "simultaneous" if the two actions begin together.
6. Return "unclear" if their order cannot be determined reliably.
7. Return only valid JSON.

Output:
{{"a_visible": true, "b_visible": true,
 "first": "A" | "B" | "simultaneous" | "unclear"}}"""

ORDER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "a_visible": {"type": "boolean"},
        "b_visible": {"type": "boolean"},
        "first": {"type": "string", "enum": ["A", "B", "simultaneous", "unclear"]},
    },
    "required": ["a_visible", "b_visible", "first"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class OrderTaskRecord:
    sample_id: str
    task_id: str
    task_family: str
    duration_seconds: float
    task_json: Path
    modalities: dict[str, Any]
    motion_groundtruth: str


@dataclass(frozen=True)
class OrderTask:
    sample_id: str
    base_sample_id: str
    task_id: str
    task_json: Path
    duration_seconds: float
    condition_text: str
    motion_groundtruth: str
    first_action: str
    second_action: str
    action_a: str
    action_b: str
    target_first: str
    video_path: Path | None
    prediction_path: Path | None
    condition_audio: str | None = None
    condition_videos: tuple[str, ...] = ()


@dataclass
class VLLMServer:
    gpu: int
    port: int
    endpoint: str
    process: subprocess.Popen
    log_path: Path
    log_handle: Any


def load_order_records(task_root: Path, modality: str = "text") -> list[OrderTaskRecord]:
    try:
        task_id_prefix = TASK_ID_PREFIXES[modality]
    except KeyError as exc:
        raise ValueError(f"unsupported Order modality: {modality}") from exc
    task_root = task_root.expanduser().resolve()
    if not task_root.is_dir():
        raise NotADirectoryError(f"Order task root does not exist: {task_root}")
    records: list[OrderTaskRecord] = []
    for path in sorted(task_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            modalities = payload["input"]["modalities"]
            ground_truth = payload["ground_truth"]
            task_id = metadata["task_id"]
            family = metadata["task_family"]
            duration = float(metadata["duration"])
            motion_groundtruth = ground_truth["motion_parameters"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Order task JSON {path}: {exc}") from exc
        if not isinstance(task_id, str) or not task_id.startswith(task_id_prefix):
            raise ValueError(f"{path}: unexpected Order task_id {task_id!r}")
        sample_id = task_id[len(task_id_prefix):]
        match = re.fullmatch(r"(.+)_p[01]", sample_id)
        if match is None:
            raise ValueError(f"{path}: Order task_id must end in _p0 or _p1")
        if not isinstance(modalities, dict):
            raise ValueError(f"{path}: input.modalities must be an object")
        if modality == "text" and not isinstance(modalities.get("text"), str):
            raise ValueError(f"{path}: input.modalities.text must be a string")
        if modality == "audio" and (
            not isinstance(modalities.get("audios"), list)
            or len(modalities["audios"]) != 1
            or not isinstance(modalities["audios"][0], str)
            or not modalities["audios"][0]
        ):
            raise ValueError(f"{path}: input.modalities.audios must contain one path")
        if modality == "video" and (
            not isinstance(modalities.get("videos_processed"), list)
            or not modalities["videos_processed"]
            or any(not isinstance(item, str) or not item for item in modalities["videos_processed"])
        ):
            raise ValueError(f"{path}: input.modalities.videos_processed must contain video paths")
        if not isinstance(motion_groundtruth, str) or not motion_groundtruth:
            raise ValueError(f"{path}: ground_truth.motion_parameters must be a path")
        if duration <= 0:
            raise ValueError(f"{path}: metadata.duration must be positive")
        records.append(OrderTaskRecord(
            sample_id=match.group(1), task_id=task_id, task_family=str(family),
            duration_seconds=duration, task_json=path, modalities=dict(modalities),
            motion_groundtruth=motion_groundtruth,
        ))
    if not records:
        raise RuntimeError(f"no Order task JSON files found under {task_root}")
    return records


def task_sample_id(record: OrderTaskRecord) -> str:
    prefix = next((value for value in TASK_ID_PREFIXES.values() if record.task_id.startswith(value)), None)
    if prefix is None:
        raise ValueError(f"{record.task_json}: unexpected Order task_id {record.task_id!r}")
    sample_id = record.task_id[len(prefix):]
    if not sample_id or not re.search(r"_p[01]$", sample_id):
        raise ValueError(f"{record.task_json}: Order task_id must end in _p0 or _p1")
    return sample_id


def parse_after_prompt(text: str) -> tuple[str, str]:
    """Return target (first, second) actions from the unambiguous p1 form."""
    match = re.fullmatch(r"do (.+) after doing (.+)", text.strip())
    if match is None:
        raise ValueError(f"invalid Order p1 prompt: {text!r}")
    second_action, first_action = (value.strip() for value in match.groups())
    if not first_action or not second_action:
        raise ValueError(f"empty action in Order p1 prompt: {text!r}")
    return first_action, second_action


def candidate_mapping(sample_id: str, first_action: str, second_action: str) -> tuple[str, str, str]:
    digest_prefix = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8]
    if int(digest_prefix, 16) % 2 == 0:
        return first_action, second_action, "A"
    return second_action, first_action, "B"


def normalize_asset_stem(stem: str) -> str:
    value = stem.removeprefix("res_")
    changed = True
    while changed:
        changed = False
        for suffix in ("_continuous", "_120"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                changed = True
    return value


def index_videos(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"rendered video root does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.startswith(".") or
                ".partial." in path.name or path.suffix.casefold() not in VIDEO_SUFFIXES):
            continue
        sample_id = normalize_asset_stem(path.stem)
        previous = result.get(sample_id)
        if previous is not None:
            if {previous.suffix.casefold(), path.suffix.casefold()} == {".mp4", ".webm"}:
                if path.suffix.casefold() == ".mp4":
                    result[sample_id] = path.resolve()
                continue
            raise ValueError(f"ambiguous rendered videos for {sample_id}: {previous}, {path}")
        result[sample_id] = path.resolve()
    return result


def index_predictions(root: Path) -> dict[str, Path]:
    from scripts.evaluation.shared.task_assets import discover_motion_clips

    result: dict[str, Path] = {}
    for raw_name, clip in discover_motion_clips(root).items():
        sample_id = normalize_asset_stem(raw_name)
        previous = result.get(sample_id)
        if previous is not None:
            raise ValueError(f"ambiguous Order predictions for {sample_id}: {previous}, {clip}")
        result[sample_id] = clip.resolve()
    return result


def build_order_tasks(
    records: Iterable[OrderTaskRecord], video_root: Path, prediction_root: Path,
    action_records: Iterable[OrderTaskRecord] | None = None,
) -> list[OrderTask]:
    action_by_sample = None
    if action_records is not None:
        action_by_sample = {task_sample_id(record): record for record in action_records}
    grouped: dict[str, dict[str, OrderTaskRecord]] = {}
    for record in records:
        if record.task_family.casefold() != "order":
            raise ValueError(f"{record.task_json}: expected metadata.task_family == 'Order'")
        sample_id = task_sample_id(record)
        variant = sample_id[-2:]
        variants = grouped.setdefault(record.sample_id, {})
        if variant in variants:
            raise ValueError(f"duplicate {variant} Order task for base sample {record.sample_id}")
        variants[variant] = record

    videos = index_videos(video_root)
    predictions = index_predictions(prediction_root)
    tasks: list[OrderTask] = []
    for base_sample_id, variants in sorted(grouped.items()):
        if set(variants) != {"p0", "p1"}:
            raise ValueError(
                f"base sample {base_sample_id} must have exactly paired p0/p1 Order tasks; "
                f"found {sorted(variants)}"
            )
        p0, p1 = variants["p0"], variants["p1"]
        label_p0, label_p1 = p0, p1
        if action_by_sample is not None:
            label_p0 = action_by_sample.get(task_sample_id(p0))
            label_p1 = action_by_sample.get(task_sample_id(p1))
            if label_p0 is None or label_p1 is None:
                raise ValueError(f"missing Text Order action labels for {base_sample_id}")
            if (label_p0.motion_groundtruth != p0.motion_groundtruth
                    or label_p1.motion_groundtruth != p1.motion_groundtruth):
                raise ValueError(f"Text/Audio Order GT mismatch for {base_sample_id}")
        first_action, second_action = parse_after_prompt(str(label_p1.modalities.get("text", "")))
        expected_p0 = f"{first_action} then {second_action}"
        expected_p1 = f"do {second_action} after doing {first_action}"
        if str(label_p0.modalities.get("text", "")).strip() != expected_p0:
            raise ValueError(
                f"{p0.task_json}: p0/p1 action identity mismatch; expected {expected_p0!r}"
            )
        if str(label_p1.modalities.get("text", "")).strip() != expected_p1:
            raise ValueError(f"{p1.task_json}: non-canonical p1 prompt")
        for record in (p0, p1):
            sample_id = task_sample_id(record)
            action_a, action_b, target_first = candidate_mapping(
                sample_id, first_action, second_action
            )
            tasks.append(OrderTask(
                sample_id=sample_id,
                base_sample_id=base_sample_id,
                task_id=record.task_id,
                task_json=record.task_json,
                duration_seconds=record.duration_seconds,
                condition_text=str((label_p0 if record is p0 else label_p1).modalities["text"]).strip(),
                motion_groundtruth=record.motion_groundtruth,
                first_action=first_action,
                second_action=second_action,
                action_a=action_a,
                action_b=action_b,
                target_first=target_first,
                video_path=videos.get(sample_id),
                prediction_path=predictions.get(sample_id),
                condition_audio=(record.modalities["audios"][0]
                                 if record.modalities.get("audios") else None),
                condition_videos=tuple(record.modalities.get("videos_processed", ())),
            ))
    return sorted(tasks, key=lambda task: task.sample_id)


def parse_order_output(raw_output: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    required = {"a_visible", "b_visible", "first"}
    if set(value) != required:
        raise ValueError(f"response fields must be exactly {sorted(required)}")
    if type(value["a_visible"]) is not bool or type(value["b_visible"]) is not bool:
        raise ValueError("a_visible and b_visible must be booleans")
    if value["first"] not in {"A", "B", "simultaneous", "unclear"}:
        raise ValueError("first is outside the allowed categorical values")
    return value


def score_order(parsed: dict[str, Any], target_first: str) -> int:
    return int(
        parsed["a_visible"] is True
        and parsed["b_visible"] is True
        and parsed["first"] == target_first
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest_sha256(model: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "tokenizer_config.json"):
        path = model / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / "records" / f"{sample_id}.json"


def _stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_key_payload(
    task: OrderTask, *, video_sha256: str, model_revision: str,
    model_manifest_sha256_value: str, fps: float, max_frames: int,
    temperature: float, seed: int,
) -> dict[str, Any]:
    return {
        "video_sha256": video_sha256,
        "task_family": "Order",
        "action_a": task.action_a,
        "action_b": task.action_b,
        "target_first": task.target_first,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "model_revision": model_revision,
        "model_manifest_sha256": model_manifest_sha256_value,
        "sampling_config": {"fps": fps, "max_frames": max_frames},
        "processor_config": {"cap_pixels_per_frame": True},
        "decoding_config": {"temperature": temperature, "do_sample": False, "seed": seed},
    }


def _cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cached_result(
    task: OrderTask, cache_dir: Path, *, model_revision: str,
    model_manifest_sha256_value: str, fps: float, max_frames: int,
    temperature: float, seed: int,
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
        video_hash = str(payload["video_sha256"])
        expected = _cache_key(_cache_key_payload(
            task, video_sha256=video_hash, model_revision=model_revision,
            model_manifest_sha256_value=model_manifest_sha256_value,
            fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
        ))
        if payload.get("cache_key") != expected:
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def invalid_asset_result(task: OrderTask, reason: str) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "sample_id": task.sample_id,
        "base_sample_id": task.base_sample_id,
        "task_family": "Order",
        "candidate_mapping": {"A": task.action_a, "B": task.action_b},
        "target_first": task.target_first,
        "valid": False,
        "decoded_video": False,
        "error": reason,
        "score": 0,
    }


def _post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        connection.request(
            "POST", f"{parsed.path.rstrip('/')}/chat/completions",
            body=body, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", errors="replace")
    finally:
        connection.close()


def infer_one(
    task: OrderTask, endpoint: str, *, cache_dir: Path, served_model_name: str,
    model_revision: str, model_manifest_sha256_value: str, fps: float,
    max_frames: int, temperature: float, seed: int, max_tokens: int,
    request_timeout: float, retries: int,
) -> dict[str, Any]:
    if task.video_path is None:
        return invalid_asset_result(task, "missing rendered prediction video")
    video_stat = _stat_signature(task.video_path)
    video_hash = sha256_file(task.video_path)
    key_payload = _cache_key_payload(
        task, video_sha256=video_hash, model_revision=model_revision,
        model_manifest_sha256_value=model_manifest_sha256_value,
        fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
    )
    request_payload = {
        "model": served_model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": task.video_path.as_uri()}},
                {"type": "text", "text": ORDER_PROMPT.format(
                    action_a=task.action_a, action_b=task.action_b
                )},
            ],
        }],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "media_io_kwargs": {"video": {"fps": fps, "num_frames": max_frames}},
        "mm_processor_kwargs": {"cap_pixels_per_frame": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "order_evaluation", "strict": True, "schema": ORDER_JSON_SCHEMA},
        },
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            status, response_body = _post_json(endpoint, request_payload, request_timeout)
            if status in {400, 404, 413, 415, 422}:
                result = invalid_asset_result(task, f"HTTP {status}: {response_body[:2000]}")
                result.update({
                    "video_path": str(task.video_path), "video_sha256": video_hash,
                    "video_stat": video_stat, "cache_key": _cache_key(key_payload),
                    "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
                    "model_revision": model_revision,
                    "model_manifest_sha256": model_manifest_sha256_value,
                    "sampling_config": {"fps": fps, "max_frames": max_frames},
                    "processor_config": {"cap_pixels_per_frame": True},
                    "decoding_config": {"temperature": temperature, "do_sample": False, "seed": seed},
                })
                _atomic_json(cache_path(cache_dir, task.sample_id), result)
                return result
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {response_body[:2000]}")
            response = json.loads(response_body)
            raw_output = response["choices"][0]["message"]["content"]
            usage = response.get("usage")
            try:
                parsed_output = parse_order_output(raw_output)
                valid = True
                error = None
                score = score_order(parsed_output, task.target_first)
            except ValueError as exc:
                parsed_output = None
                valid = False
                error = str(exc)
                score = 0
            result = {
                "schema": CACHE_SCHEMA,
                "sample_id": task.sample_id,
                "base_sample_id": task.base_sample_id,
                "video_path": str(task.video_path),
                "video_sha256": video_hash,
                "video_stat": video_stat,
                "task_family": "Order",
                "candidate_mapping": {"A": task.action_a, "B": task.action_b},
                "target_first": task.target_first,
                "prompt_version": PROMPT_VERSION,
                "parser_version": PARSER_VERSION,
                "model_revision": model_revision,
                "model_manifest_sha256": model_manifest_sha256_value,
                "sampling_config": {"fps": fps, "max_frames": max_frames},
                "processor_config": {"cap_pixels_per_frame": True},
                "decoding_config": {"temperature": temperature, "do_sample": False, "seed": seed},
                "endpoint": endpoint,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "valid": valid,
                "decoded_video": True,
                "error": error,
                "score": score,
                "usage": usage,
                "cache_key": _cache_key(key_payload),
            }
            _atomic_json(cache_path(cache_dir, task.sample_id), result)
            return result
        except (OSError, TimeoutError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"infrastructure request failure for {task.sample_id} via {endpoint}: {last_error}")


def _health(endpoint: str, timeout: float = 2.0) -> bool:
    parsed = urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("GET", f"{parsed.path.rstrip('/')}/models")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        connection.close()


def start_vllm_servers(
    *, model: Path, gpus: list[int], base_port: int, served_model_name: str,
    video_root: Path, log_dir: Path, max_model_len: int, max_num_seqs: int,
    gpu_memory_utilization: float, startup_timeout: float,
    max_num_batched_tokens: int | None = None,
) -> list[VLLMServer]:
    executable = shutil.which("vllm")
    if executable is None:
        sibling = Path(sys.executable).resolve().parent / "vllm"
        if sibling.is_file() and os.access(sibling, os.X_OK):
            executable = str(sibling)
    if executable is None:
        raise RuntimeError("vllm executable is not available in the active environment")
    log_dir.mkdir(parents=True, exist_ok=True)
    servers: list[VLLMServer] = []
    try:
        for offset, gpu in enumerate(gpus):
            port = base_port + offset
            endpoint = f"http://127.0.0.1:{port}/v1"
            log_path = log_dir / f"vllm_gpu{gpu}_port{port}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "OMP_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            })
            command = [
                executable, "serve", str(model),
                "--served-model-name", served_model_name,
                "--host", "127.0.0.1", "--port", str(port),
                "--tensor-parallel-size", "1",
                "--max-model-len", str(max_model_len),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-num-seqs", str(max_num_seqs),
                "--limit-mm-per-prompt", '{"video":1,"image":0}',
                "--mm-processor-kwargs", '{"cap_pixels_per_frame":true}',
                "--allowed-local-media-path", str(video_root.resolve()),
                "--seed", "0",
            ]
            if max_num_batched_tokens is not None:
                command.extend(("--max-num-batched-tokens", str(max_num_batched_tokens)))
            process = subprocess.Popen(
                command, stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,
            )
            servers.append(VLLMServer(gpu, port, endpoint, process, log_path, log_handle))

        deadline = time.monotonic() + startup_timeout
        pending = set(range(len(servers)))
        while pending and time.monotonic() < deadline:
            for index in list(pending):
                server = servers[index]
                if server.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server on GPU {server.gpu} exited with code "
                        f"{server.process.returncode}; see {server.log_path}"
                    )
                if _health(server.endpoint):
                    pending.remove(index)
            if pending:
                time.sleep(2)
        if pending:
            waiting = [f"GPU {servers[index].gpu} ({servers[index].log_path})" for index in pending]
            raise TimeoutError(f"vLLM startup timed out: {', '.join(waiting)}")
        return servers
    except BaseException:
        stop_vllm_servers(servers)
        raise


def stop_vllm_servers(servers: Iterable[VLLMServer]) -> None:
    servers = list(servers)
    for server in servers:
        if server.process.poll() is None:
            try:
                os.killpg(server.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 45
    for server in servers:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            server.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for server in servers:
        if server.process.poll() is None:
            try:
                os.killpg(server.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.process.wait(timeout=10)
        server.log_handle.close()


def run_inference(
    tasks: list[OrderTask], endpoints: list[str], *, cache_dir: Path,
    served_model_name: str, model_revision: str, model_manifest_sha256_value: str,
    fps: float, max_frames: int, temperature: float, seed: int,
    max_tokens: int, request_timeout: float, retries: int,
    requests_per_server: int, progress_path: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pending: list[OrderTask] = []
    for task in tasks:
        if task.video_path is None:
            results[task.sample_id] = invalid_asset_result(task, "missing rendered prediction video")
            continue
        cached = load_cached_result(
            task, cache_dir, model_revision=model_revision,
            model_manifest_sha256_value=model_manifest_sha256_value,
            fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
        )
        if cached is None:
            pending.append(task)
        else:
            results[task.sample_id] = cached

    if pending and not endpoints:
        raise RuntimeError(f"{len(pending)} uncached Order tasks remain but no vLLM endpoint is available")
    total = len(tasks)
    started = time.monotonic()
    _atomic_json(progress_path, {
        "expected": total, "cached": len(results), "pending": len(pending),
        "completed": len(results), "status": "running", "updated_at": time.time(),
    })
    if not pending:
        return results

    semaphores = {endpoint: threading.Semaphore(requests_per_server) for endpoint in endpoints}

    def execute(task: OrderTask, endpoint: str) -> dict[str, Any]:
        with semaphores[endpoint]:
            return infer_one(
                task, endpoint, cache_dir=cache_dir, served_model_name=served_model_name,
                model_revision=model_revision,
                model_manifest_sha256_value=model_manifest_sha256_value,
                fps=fps, max_frames=max_frames, temperature=temperature, seed=seed,
                max_tokens=max_tokens, request_timeout=request_timeout, retries=retries,
            )

    workers = len(endpoints) * requests_per_server
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="order-vllm") as executor:
        future_to_task = {}
        for index, task in enumerate(pending):
            endpoint = endpoints[index % len(endpoints)]
            future_to_task[executor.submit(execute, task, endpoint)] = task
        completed_new = 0
        last_report = time.monotonic()
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results[task.sample_id] = result
            completed_new += 1
            now = time.monotonic()
            if completed_new % 25 == 0 or now - last_report >= 30 or completed_new == len(pending):
                completed = len(results)
                elapsed = max(now - started, 1e-6)
                rate = completed_new / elapsed
                eta = (len(pending) - completed_new) / rate if rate > 0 else None
                progress = {
                    "expected": total, "cached_at_start": total - len(pending),
                    "pending_at_start": len(pending), "completed": completed,
                    "completed_new": completed_new, "rate_samples_per_second": rate,
                    "eta_seconds": eta, "status": "running", "updated_at": time.time(),
                }
                _atomic_json(progress_path, progress)
                print(
                    f"Order VideoLLM progress: {completed}/{total} "
                    f"({rate:.2f} new samples/s, ETA {eta / 60:.1f} min)" if eta is not None
                    else f"Order VideoLLM progress: {completed}/{total}",
                    flush=True,
                )
                last_report = now
    _atomic_json(progress_path, {
        "expected": total, "completed": len(results), "status": "complete",
        "elapsed_seconds": time.monotonic() - started, "updated_at": time.time(),
    })
    return results
