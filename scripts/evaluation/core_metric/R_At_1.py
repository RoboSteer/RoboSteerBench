"""Formal fixed-candidate retrieval metrics for paired X--motion embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    """Ranks and recall values for deterministic complete fixed-size groups."""

    ranks: np.ndarray
    hits_at_1: np.ndarray
    num_input_pairs: int
    num_used_pairs: int
    batch_size: int
    order: np.ndarray
    excluded_indices: np.ndarray
    order_policy: str

    @property
    def r_at_1(self) -> float:
        """Backward-compatible shorthand for ``recall_at(1)``."""
        return self.recall_at(1)

    def hits_at(self, top_k: int) -> np.ndarray:
        """Return one hit indicator per query for fixed-candidate Recall@K."""
        if not 1 <= top_k <= self.batch_size:
            raise ValueError(
                f"top_k must be in [1, {self.batch_size}], got {top_k}"
            )
        return (self.ranks <= top_k).astype(np.float32)

    def recall_at(self, top_k: int) -> float:
        """Mean fixed-candidate Recall@K over all complete candidate groups."""
        hits = self.hits_at(top_k)
        if hits.size == 0:
            raise ValueError(f"R@{top_k} requires at least one complete retrieval batch")
        return float(hits.mean())


def fixed_batch_retrieval(
    condition_embeddings: np.ndarray,
    motion_embeddings: np.ndarray,
    *,
    batch_size: int = 32,
    dataset_names: list[str] | None = None,
    seed: int = 0,
) -> RetrievalResult:
    """OMG non-strict fixed-candidate Motion-to-Condition retrieval.

    Samples are optionally stratified by dataset with a fixed seed, then only
    the leading complete fixed-size candidate groups are evaluated. This is
    the official non-strict ``require_full_batches=False`` behavior.
    """
    condition = np.asarray(condition_embeddings, dtype=np.float64)
    motion = np.asarray(motion_embeddings, dtype=np.float64)
    if condition.ndim != 2 or motion.ndim != 2 or condition.shape != motion.shape:
        raise ValueError(
            "retrieval requires finite, identically shaped (N, D) condition and motion arrays; "
            f"got {condition.shape} and {motion.shape}"
        )
    if condition.shape[0] == 0 or not np.isfinite(condition).all() or not np.isfinite(motion).all():
        raise ValueError("retrieval requires non-empty finite embeddings")
    if batch_size <= 1:
        raise ValueError("retrieval batch_size must be greater than one")
    count = int(condition.shape[0])
    if dataset_names is not None and len(dataset_names) != count:
        raise ValueError("dataset_names length must equal embedding count")
    usable = (count // batch_size) * batch_size
    if usable == 0:
        raise ValueError(f"R@K needs at least one complete fixed-{batch_size} candidate group, got {count}")
    full_order = (
        _stratified_order(dataset_names, count, batch_size, seed)
        if dataset_names is not None else np.arange(count, dtype=np.int64)
    )
    order = full_order[:usable]
    excluded = np.sort(full_order[usable:])
    order_policy = "stratified_by_dataset" if dataset_names is not None else "manifest_order"
    ranks: list[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        if len(indices) != batch_size:
            raise RuntimeError("retrieval grouping produced an incomplete candidate batch")
        cond_batch = condition[indices]
        motion_batch = motion[indices]
        distances_sq = (
            np.square(motion_batch).sum(axis=1, keepdims=True)
            - 2.0 * motion_batch.dot(cond_batch.T)
            + np.square(cond_batch).sum(axis=1)[None, :]
        )
        matched = np.diag(distances_sq)
        ranks.append((distances_sq < matched[:, None]).sum(axis=1).astype(np.int64) + 1)
    rank_values = np.concatenate(ranks)
    return RetrievalResult(
        ranks=rank_values,
        hits_at_1=(rank_values == 1).astype(np.float32),
        num_input_pairs=count,
        num_used_pairs=len(order),
        batch_size=int(batch_size),
        order=order,
        excluded_indices=excluded,
        order_policy=order_policy,
    )




def _stratified_order(dataset_names: list[str], count: int, group_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = {}
    for index, name in enumerate(dataset_names):
        buckets.setdefault(str(name), []).append(index)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    cursors = {key: 0 for key in keys}
    rotate = 0
    order: list[int] = []
    while len(order) < count:
        active = [key for key in keys if cursors[key] < len(buckets[key])]
        if not active:
            break
        rotated = active[rotate % len(active):] + active[:rotate % len(active)]
        batch: list[int] = []
        while len(batch) < group_size and active:
            progressed = False
            for key in rotated:
                if cursors[key] < len(buckets[key]):
                    batch.append(buckets[key][cursors[key]])
                    cursors[key] += 1
                    progressed = True
                    if len(batch) == group_size:
                        break
            if not progressed:
                break
            active = [key for key in keys if cursors[key] < len(buckets[key])]
            rotated = active[rotate % len(active):] + active[:rotate % len(active)] if active else []
        order.extend(batch)
        rotate += 1
    return np.asarray(order, dtype=np.int64)
