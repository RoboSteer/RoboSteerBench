from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np
import torch


def paired_mm_distance(audio_embeddings: np.ndarray, motion_embeddings: np.ndarray) -> float:
    audio = torch.as_tensor(np.asarray(audio_embeddings), dtype=torch.float64)
    motion = torch.as_tensor(np.asarray(motion_embeddings), dtype=torch.float64)
    if audio.shape != motion.shape or audio.ndim != 2:
        raise ValueError("Expected paired 2D embedding arrays with identical shapes")
    return float(torch.linalg.vector_norm(audio - motion, dim=1).mean())


def r_precision(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray, top_k: int = 3) -> np.ndarray:
    query = torch.as_tensor(np.asarray(query_embeddings), dtype=torch.float64)
    candidates = torch.as_tensor(np.asarray(candidate_embeddings), dtype=torch.float64)
    if query.shape != candidates.shape or query.ndim != 2:
        raise ValueError("Expected paired 2D embedding arrays with identical shapes")
    distances = (
        -2.0 * (query @ candidates.T)
        + torch.square(query).sum(dim=1, keepdim=True)
        + torch.square(candidates).sum(dim=1)[None, :]
    )
    paired = torch.diagonal(distances)
    ranks = (distances < paired[:, None]).sum(dim=1) + 1
    return torch.stack([(ranks <= k).to(torch.float64).mean() for k in range(1, top_k + 1)]).cpu().numpy()


def _stratified_order(dataset_names: list[str], count: int, group_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(dataset_names):
        groups.setdefault(str(name), []).append(index)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    cursors = {key: 0 for key in keys}
    order: list[int] = []
    rotate = 0
    while len(order) < count:
        active = [key for key in keys if cursors[key] < len(groups[key])]
        rotated = active[rotate % len(active) :] + active[: rotate % len(active)] if active else []
        batch: list[int] = []
        while len(batch) < group_size and active:
            progressed = False
            for key in rotated:
                if cursors[key] < len(groups[key]):
                    batch.append(groups[key][cursors[key]])
                    cursors[key] += 1
                    progressed = True
                    if len(batch) == group_size:
                        break
            if not progressed:
                break
            active = [key for key in keys if cursors[key] < len(groups[key])]
            rotated = active[rotate % len(active) :] + active[: rotate % len(active)] if active else []
        order.extend(batch)
        rotate += 1
    return np.asarray(order, dtype=np.int64)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"semantic-group-v1\0{seed}\0{value}".encode("utf-8")).hexdigest()


def semantic_aware_group_order(
    semantic_ids: list[str],
    group_size: int,
    seed: int,
    *,
    dataset_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build full fixed-size groups with no repeated semantic ID per group."""
    count = len(semantic_ids)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    num_groups = count // group_size
    if num_groups == 0:
        raise ValueError(f"Need at least {group_size} samples for one retrieval group")
    if dataset_names is not None and len(dataset_names) != count:
        raise ValueError("dataset_names length must equal semantic_ids length")
    target_count = num_groups * group_size
    exclusion_budget = count - target_count

    buckets: dict[str, list[int]] = {}
    for index, semantic_id in enumerate(semantic_ids):
        buckets.setdefault(str(semantic_id), []).append(index)
    for semantic_id, indices in buckets.items():
        indices.sort(key=lambda index: (_stable_key(seed, f"item:{semantic_id}:{index}"), index))

    excluded: list[int] = []
    # A semantic can occur at most once in each output group.
    mandatory_exclusions = sum(max(0, len(indices) - num_groups) for indices in buckets.values())
    if mandatory_exclusions > exclusion_budget:
        raise ValueError(
            "Cannot construct semantic-conflict-free fixed groups: "
            f"need at least {mandatory_exclusions} exclusions but only {exclusion_budget} remainder slots exist"
        )
    for semantic_id in sorted(buckets):
        indices = buckets[semantic_id]
        excess = max(0, len(indices) - num_groups)
        if excess:
            excluded.extend(indices[-excess:])
            buckets[semantic_id] = indices[:-excess]

    # Deterministically spend the remaining remainder on the most redundant semantics.
    while len(excluded) < exclusion_budget:
        candidates = [(len(indices), semantic_id) for semantic_id, indices in buckets.items() if indices]
        if not candidates:
            raise ValueError("Cannot select the required semantic-aware validation subset")
        largest = max(size for size, _ in candidates)
        tied = [semantic_id for size, semantic_id in candidates if size == largest]
        semantic_id = min(tied, key=lambda value: (_stable_key(seed, f"exclude:{value}"), value))
        excluded.append(buckets[semantic_id].pop())

    output_groups: list[list[int]] = [[] for _ in range(num_groups)]
    output_semantics: list[set[str]] = [set() for _ in range(num_groups)]
    output_dataset_counts: list[Counter[str]] = [Counter() for _ in range(num_groups)]
    ordered_buckets = sorted(
        ((semantic_id, indices) for semantic_id, indices in buckets.items() if indices),
        key=lambda item: (-len(item[1]), _stable_key(seed, f"semantic:{item[0]}"), item[0]),
    )
    for semantic_id, indices in ordered_buckets:
        candidate_groups = sorted(
            range(num_groups),
            key=lambda group: (
                len(output_groups[group]),
                _stable_key(seed, f"group:{semantic_id}:{group}"),
                group,
            ),
        )[: len(indices)]
        if len(candidate_groups) != len(indices):
            raise ValueError(f"Cannot place semantic group {semantic_id!r} without a collision")
        remaining_groups = set(candidate_groups)
        for index in indices:
            dataset_name = str(dataset_names[index]) if dataset_names is not None else ""
            group = min(
                remaining_groups,
                key=lambda candidate: (
                    output_dataset_counts[candidate][dataset_name],
                    _stable_key(seed, f"place:{semantic_id}:{index}:{candidate}"),
                    candidate,
                ),
            )
            output_groups[group].append(index)
            output_semantics[group].add(semantic_id)
            output_dataset_counts[group][dataset_name] += 1
            remaining_groups.remove(group)

    sizes = [len(group) for group in output_groups]
    if any(size != group_size for size in sizes):
        raise ValueError(f"Semantic-aware grouping failed to fill fixed groups: sizes={sizes}")
    for group, semantics in zip(output_groups, output_semantics):
        if len(group) != len(semantics):
            raise ValueError("Semantic-aware grouping produced a duplicate semantic ID")
    order = np.asarray([index for group in output_groups for index in group], dtype=np.int64)
    excluded_array = np.asarray(sorted(excluded), dtype=np.int64)
    if len(order) != target_count or len(set(order.tolist()) & set(excluded_array.tolist())):
        raise ValueError("Semantic-aware grouping produced inconsistent selected/excluded indices")
    return order, excluded_array


def evaluate_fixed_groups(
    motion_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    group_size: int = 32,
    *,
    dataset_names: list[str] | None = None,
    semantic_ids: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    motion = np.asarray(motion_embeddings)
    audio = np.asarray(audio_embeddings)
    if audio.shape != motion.shape:
        raise ValueError("Embedding arrays must have identical shapes")
    if len(audio) < group_size:
        raise ValueError(f"Official protocol needs at least one complete group of {group_size}")
    if dataset_names is not None and len(dataset_names) != len(audio):
        raise ValueError("dataset_names length must equal embedding count")
    if semantic_ids is not None:
        if len(semantic_ids) != len(audio):
            raise ValueError("semantic_ids length must equal embedding count")
        order, excluded = semantic_aware_group_order(
            semantic_ids,
            group_size,
            seed,
            dataset_names=dataset_names,
        )
        order_policy = "semantic_unique_then_dataset_balanced"
    else:
        if len(audio) % group_size:
            raise ValueError(f"Official fixed-{group_size} protocol requires a divisible pair count, got {len(audio)}")
        order = (
            _stratified_order(dataset_names, len(audio), group_size, seed)
            if dataset_names is not None
            else np.arange(len(audio), dtype=np.int64)
        )
        excluded = np.asarray([], dtype=np.int64)
        order_policy = "stratified_by_dataset" if dataset_names is not None else "manifest_order"
    r_values = []
    semantic_conflicts = 0
    for start in range(0, len(order), group_size):
        end = start + group_size
        indices = order[start:end]
        if semantic_ids is not None:
            group_semantics = [semantic_ids[int(index)] for index in indices]
            semantic_conflicts += len(group_semantics) - len(set(group_semantics))
        # Exact OMG direction: motion rows query the paired modality columns.
        r_values.append(r_precision(motion[indices], audio[indices], top_k=3))
    if semantic_conflicts:
        raise ValueError(f"Semantic-aware retrieval groups contain {semantic_conflicts} conflicts")
    mean_r = np.stack(r_values).mean(axis=0)
    return {
        "protocol": "OMG fixed candidate groups, strict-less-than rank, Motion-to-Audio",
        "group_size": group_size,
        "pool_pairs": len(audio),
        "evaluated_pairs": len(order),
        "excluded_remainder": len(excluded),
        "retrieval_num_groups": len(order) // group_size,
        "retrieval_order_policy": order_policy,
        "seed": seed,
        "retrieval_order_indices": order.tolist(),
        "excluded_indices": excluded.tolist(),
        "semantic_conflicts": semantic_conflicts,
        "r_at_1": float(mean_r[0]),
        "r_at_2": float(mean_r[1]) if len(mean_r) > 1 else None,
        "r_at_3": float(mean_r[2]) if len(mean_r) > 2 else None,
        "mm_distance": paired_mm_distance(motion[order], audio[order]),
    }
