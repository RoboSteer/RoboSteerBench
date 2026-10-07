from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audio_motion_evaluator.inference import AudioMotionEvaluator
from audio_motion_evaluator.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from audio_motion_evaluator.metrics import evaluate_fixed_groups


GROUP_SIZE = 32


def _normalise_semantic_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _semantic_identity(record: dict[str, Any]) -> tuple[str, str]:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    explicit = record.get("semantic_id") or meta.get("semantic_id")
    if explicit:
        return str(explicit), "explicit_semantic_id"
    for key, owner in (
        ("caption", record),
        ("action_text", meta),
        ("transcript", meta),
        ("text", meta),
    ):
        text = _normalise_semantic_text(str(owner.get(key, "")))
        if text:
            digest = hashlib.sha256(f"semantic-text-v1\0{text}".encode("utf-8")).hexdigest()
            return digest, key
    raise ValueError(
        "Every sample needs semantic_id or non-empty caption/action_text/transcript/text. "
        "source_id alone is not sufficient to detect duplicate semantics."
    )


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    required = {"qpos_36", "fps", "dataset", "dataset_index"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"{path} is missing artifact keys: {missing}")
    qpos = np.asarray(arrays["qpos_36"])
    if qpos.ndim != 3 or qpos.shape[-1] != 36:
        raise ValueError(f"qpos_36 must have shape (N,T,36), got {qpos.shape}")
    if np.asarray(arrays["fps"]).shape != (len(qpos),):
        raise ValueError("fps must have one value per sample")
    if "valid_lengths" in arrays:
        lengths = np.asarray(arrays["valid_lengths"])
        if lengths.shape != (len(qpos),):
            raise ValueError("valid_lengths must have one value per sample")
        if np.any(lengths < 2) or np.any(lengths > qpos.shape[1]):
            raise ValueError("valid_lengths must be between 2 and the padded qpos frame count")
    if "valid" in arrays:
        valid = np.asarray(arrays["valid"])
        if valid.shape != qpos.shape[:2]:
            raise ValueError("valid mask shape must match qpos_36[:2]")
    return arrays


def _valid_qpos(arrays: dict[str, np.ndarray], index: int) -> np.ndarray:
    qpos = arrays["qpos_36"][index]
    if "valid_lengths" in arrays:
        length = int(arrays["valid_lengths"][index])
    elif "valid" in arrays:
        mask = np.asarray(arrays["valid"][index], dtype=np.bool_)
        length = int(mask.sum())
        if not np.array_equal(mask, np.arange(len(mask)) < length):
            raise ValueError(f"Sample {index} valid mask must be a contiguous prefix")
    else:
        length = len(qpos)
    return qpos[:length]


def _audio_path(record: dict[str, Any], samples_path: Path) -> Path:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    raw = record.get("audio_path") or meta.get("audio_path")
    if not raw:
        raise ValueError("Every sample must provide audio_path or meta.audio_path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = samples_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    return path


def _sample_identity(record: dict[str, Any], index: int) -> str:
    return str(record.get("source_id") or record.get("pair_id") or f"sample-{index:06d}")


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cache_path(cache_dir: Path, index: int, source_id: str) -> Path:
    suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{index:06d}_{suffix}.npz"


def _read_cache(path: Path, expected: dict[str, str], need_reference: bool) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as handle:
            for key, value in expected.items():
                if key not in handle or str(handle[key]) != value:
                    return None
            required = {"audio_embedding", "generated_motion_embedding"}
            if need_reference:
                required.add("reference_motion_embedding")
            if not required.issubset(handle.files):
                return None
            return {key: handle[key].astype(np.float32, copy=True) for key in required}
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate speech-to-motion semantic retrieval on an OMG qpos artifact."
    )
    parser.add_argument("--generated-qpos", type=Path, required=True)
    parser.add_argument("--reference-qpos", type=Path)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--omg-root", type=Path, required=True)
    parser.add_argument("--omg-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generated_path = args.generated_qpos.expanduser().resolve()
    samples_path = args.samples_path.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = _load_artifact(generated_path)
    reference = _load_artifact(args.reference_qpos.expanduser().resolve()) if args.reference_qpos else None
    records = read_jsonl(samples_path)
    count = len(generated["qpos_36"])
    if len(records) != count:
        raise ValueError(f"samples count {len(records)} != generated motion count {count}")
    if reference is not None and len(reference["qpos_36"]) != count:
        raise ValueError("reference and generated artifact sample counts differ")
    if count < GROUP_SIZE:
        raise ValueError(f"OMG fixed-batch32 retrieval requires at least {GROUP_SIZE} samples")

    resolved: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        artifact_dataset = str(generated["dataset"][index])
        artifact_index = int(generated["dataset_index"][index])
        record_dataset = str(record.get("dataset", artifact_dataset))
        record_index = int(record.get("index", artifact_index))
        if record_dataset != artifact_dataset or record_index != artifact_index:
            raise ValueError(
                "samples_path order does not match generated_qpos at "
                f"sample {index}: samples=({record_dataset},{record_index}) "
                f"artifact=({artifact_dataset},{artifact_index})"
            )
        semantic_id, semantic_source = _semantic_identity(record)
        audio_path = _audio_path(record, samples_path)
        resolved.append(
            {
                "index": index,
                "source_id": _sample_identity(record, index),
                "dataset": record_dataset,
                "dataset_index": record_index,
                "audio_path": str(audio_path),
                "audio_sha256": sha256_file(audio_path),
                "semantic_id": semantic_id,
                "semantic_id_source": semantic_source,
            }
        )

    checkpoint_sha = sha256_file(checkpoint_path)
    cache_dir = output_dir / "embedding_cache"
    need_reference = reference is not None
    cache_entries: list[dict[str, np.ndarray] | None] = []
    expected_per_sample: list[dict[str, str]] = []
    missing = 0
    for index, row in enumerate(resolved):
        expected = {
            "checkpoint_sha256": checkpoint_sha,
            "audio_sha256": row["audio_sha256"],
            "generated_motion_sha256": _array_hash(_valid_qpos(generated, index)),
        }
        if need_reference:
            expected["reference_motion_sha256"] = _array_hash(_valid_qpos(reference, index))
        expected_per_sample.append(expected)
        cached = _read_cache(
            _cache_path(cache_dir, index, row["source_id"]), expected, need_reference
        )
        cache_entries.append(cached)
        missing += cached is None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[artifact] pairs={count} fixed_group={GROUP_SIZE} cached={count - missing} "
        f"to_compute={missing} device={device}",
        flush=True,
    )
    evaluator = None
    if missing:
        evaluator = AudioMotionEvaluator(
            checkpoint_path,
            omg_root=args.omg_root,
            omg_checkpoint=args.omg_checkpoint,
            device=device,
        )
    for index in tqdm(range(count), desc="encode artifact", unit="pair"):
        if cache_entries[index] is not None:
            continue
        assert evaluator is not None
        row = resolved[index]
        audio_embedding, audio_metadata = evaluator.encode_audio(Path(row["audio_path"]))
        generated_embedding, generated_metadata = evaluator.encode_motion(
            _valid_qpos(generated, index), float(generated["fps"][index])
        )
        values: dict[str, Any] = {
            **{key: np.asarray(value) for key, value in expected_per_sample[index].items()},
            "audio_embedding": audio_embedding,
            "generated_motion_embedding": generated_embedding,
            "audio_preprocessing_json": np.asarray(json.dumps(audio_metadata, sort_keys=True)),
            "generated_motion_preprocessing_json": np.asarray(json.dumps(generated_metadata, sort_keys=True)),
        }
        if need_reference:
            reference_embedding, reference_metadata = evaluator.encode_motion(
                _valid_qpos(reference, index), float(reference["fps"][index])
            )
            values["reference_motion_embedding"] = reference_embedding
            values["reference_motion_preprocessing_json"] = np.asarray(
                json.dumps(reference_metadata, sort_keys=True)
            )
        path = _cache_path(cache_dir, index, row["source_id"])
        _write_cache(path, values)
        cache_entries[index] = _read_cache(path, expected_per_sample[index], need_reference)
        if cache_entries[index] is None:
            raise RuntimeError(f"Could not verify cache immediately after writing: {path}")

    audio_embeddings = np.stack([entry["audio_embedding"] for entry in cache_entries if entry])
    generated_embeddings = np.stack(
        [entry["generated_motion_embedding"] for entry in cache_entries if entry]
    )
    semantic_ids = [row["semantic_id"] for row in resolved]
    datasets = [row["dataset"] for row in resolved]
    generated_metrics = evaluate_fixed_groups(
        generated_embeddings,
        audio_embeddings,
        group_size=GROUP_SIZE,
        semantic_ids=semantic_ids,
        dataset_names=datasets,
        seed=args.seed,
    )
    metrics: dict[str, Any] = {
        "schema": "audio_motion_artifact_evaluation.v1",
        "generated": generated_metrics,
    }
    if need_reference:
        reference_embeddings = np.stack(
            [entry["reference_motion_embedding"] for entry in cache_entries if entry]
        )
        metrics["reference"] = evaluate_fixed_groups(
            reference_embeddings,
            audio_embeddings,
            group_size=GROUP_SIZE,
            semantic_ids=semantic_ids,
            dataset_names=datasets,
            seed=args.seed,
        )
    else:
        reference_embeddings = None

    selected = generated_metrics["retrieval_order_indices"]
    excluded = generated_metrics["excluded_indices"]
    metrics["provenance"] = {
        "generated_qpos": str(generated_path),
        "generated_qpos_sha256": sha256_file(generated_path),
        "reference_qpos": str(args.reference_qpos.expanduser().resolve()) if args.reference_qpos else None,
        "reference_qpos_sha256": sha256_file(args.reference_qpos.expanduser().resolve()) if args.reference_qpos else None,
        "samples_path": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "evaluator_checkpoint": str(checkpoint_path),
        "evaluator_checkpoint_sha256": checkpoint_sha,
        "omg_motion_checkpoint": str(args.omg_checkpoint.expanduser().resolve()),
        "omg_motion_checkpoint_sha256": sha256_file(args.omg_checkpoint.expanduser().resolve()),
        "device": str(device),
        "fixed_group_seed": args.seed,
        "retrieval_order_source_ids": [resolved[index]["source_id"] for index in selected],
        "excluded_source_ids": [resolved[index]["source_id"] for index in excluded],
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_jsonl(output_dir / "resolved_samples.jsonl", resolved)
    arrays: dict[str, Any] = {
        "audio_embeddings": audio_embeddings.astype(np.float32),
        "generated_motion_embeddings": generated_embeddings.astype(np.float32),
        "semantic_ids": np.asarray(semantic_ids, dtype=np.str_),
    }
    if reference_embeddings is not None:
        arrays["reference_motion_embeddings"] = reference_embeddings.astype(np.float32)
    np.savez_compressed(output_dir / "embeddings.npz", **arrays)
    lines = [
        "# Audio-Motion semantic evaluation",
        "",
        f"- Protocol: OMG fixed-batch{GROUP_SIZE}, Motion→Audio, strict diagonal matching",
        f"- Evaluated pairs: {generated_metrics['evaluated_pairs']} / {count}",
        f"- Semantic conflicts: {generated_metrics['semantic_conflicts']}",
        f"- Generated R@1: {generated_metrics['r_at_1']:.6f}",
        f"- Generated R@2: {generated_metrics['r_at_2']:.6f}",
        f"- Generated R@3: {generated_metrics['r_at_3']:.6f}",
        f"- Generated MM Distance: {generated_metrics['mm_distance']:.6f}",
    ]
    if "reference" in metrics:
        lines.extend(
            [
                f"- Reference R@1: {metrics['reference']['r_at_1']:.6f}",
                f"- Reference MM Distance: {metrics['reference']['mm_distance']:.6f}",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
