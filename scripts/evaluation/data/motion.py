"""Discover motion clips and load the common G1 CSV representation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


REQUIRED_CSV = ("body_pos.csv", "body_quat.csv", "joint_pos.csv")
LDA_SUFFIX = "_music_unitree_g1_headless"
AUDIOMOTION_SUFFIX = "_audio"
UH1_PREDICTION_SUFFIX = "_pred"
UH1_CONTINUOUS_SUFFIX = "_continuous"
VIDEO2ROBOT_HUMAN_SUFFIX = "_HUMAN"
VIDEO2ROBOT_SKELETON_SUFFIX = "_SKEL"
TEMPORAL_COMPLETION_GROUNDTRUTH_SUFFIX = "_fore"
TEMPORAL_COMPLETION_SKELETON_INPUT_SUFFIX = "_fore_skel_input"
TEMPORAL_RETROSPECTION_GROUNDTRUTH_SUFFIX = "_retro"
TEMPORAL_RETROSPECTION_SKELETON_INPUT_SUFFIX = "_retro_skel_input"
TEMPORAL_INTERPOLATION_GROUNDTRUTH_SUFFIX = "_inter"
SPATIAL_COMPLETION_PREDICTION_SUFFIXES = (
    "_no_upper_skel", "_no_lower_skel",
    "_no_upper", "_no_lower",
)
MOTIONCRAFT_PREFIX = "res_"
MOTIONCRAFT_SUFFIX_MARKER = "_music_"
TEXT_RESULT_FRAME_SUFFIX = "_120"
BFM_ZERO_LEAF_DIRECTORY = "sonic"
LOM_TEXT_PREFIXES = (
    "L1_IMG_TXT_SKEL_",
    "L1_IMG_TXT_HUMAN_",
    "L1_TXT_GEN_",
    "L1_TXT_COMP_HANDS_",
    "L1_TXT_COMP_LEGS_",
    "L1_TXT_FORE_",
    "L1_TXT_INTER_",
    "L1_TXT_RETRO_",
    "L1_MUL_BAL_",
    # MotionCraft fixed-length Level-2 text exports use the same wrapper as
    # Level-1: ``res_<task-prefix><source-id>_120``. Keep the task-specific
    # suffix (for example ``_fast``/``_slow``) because Level-2 uses it to
    # resolve the matching task JSON.
    "L2_SPEED_TXT_",
    "L2_AMPLITUDE_TXT_",
    "L2_BODYRESTRAIN_TXT_",
    "L2_DIR_TXT_",
    "L2_DIRECTION_TXT_",
    "L2_TRAJ_TXT_",
    "L2_TRAJECTORY_TXT_",
)


@dataclass(frozen=True)
class MotionIndex:
    root: Path
    samples: dict[str, Path]
    duplicates: dict[str, list[Path]]


def normalize_sample_id(name: str) -> str:
    """Map known model rollout directory names to the source motion ID.

    Audio-motion exports append ``_audio`` to the otherwise canonical OMG
    sample ID. The suffix identifies the rollout modality, not the task, so
    remove it before matching task metadata or motion ground truth.
    """
    sample_id = name
    if sample_id.endswith(LDA_SUFFIX):
        sample_id = sample_id[: -len(LDA_SUFFIX)]
    # MotionCraft emits: res_<source_id>_music_<seed>_<rank>_epoch_<epoch>.
    # Keep exactly the source ID used by motion ground truth.
    elif sample_id.startswith(MOTIONCRAFT_PREFIX) and MOTIONCRAFT_SUFFIX_MARKER in sample_id:
        sample_id = sample_id[len(MOTIONCRAFT_PREFIX):].split(MOTIONCRAFT_SUFFIX_MARKER, 1)[0]
    # Some fixed-length Text task exports wrap the task ID as
    # ``res_L<level>_<TASK>_<source_id>_120``. Remove only this fully
    # recognized wrapper; a generic numeric-suffix removal could corrupt valid
    # source IDs.
    elif sample_id.startswith(MOTIONCRAFT_PREFIX) and sample_id.endswith(TEXT_RESULT_FRAME_SUFFIX):
        unwrapped = sample_id[len(MOTIONCRAFT_PREFIX) : -len(TEXT_RESULT_FRAME_SUFFIX)]
        if any(unwrapped.startswith(prefix) for prefix in LOM_TEXT_PREFIXES):
            sample_id = unwrapped
    for prefix in LOM_TEXT_PREFIXES:
        if sample_id.startswith(prefix):
            sample_id = sample_id[len(prefix):]
            break
    # Temporal task skeleton inputs retain both the task and representation markers;
    # neither is part of the shared source sample ID.
    sample_id = sample_id.removesuffix(TEMPORAL_COMPLETION_SKELETON_INPUT_SUFFIX)
    sample_id = sample_id.removesuffix(TEMPORAL_RETROSPECTION_SKELETON_INPUT_SUFFIX)
    # Video2Robot exports use the representation suffix; it is not part of the shared source ID.
    sample_id = sample_id.removesuffix(VIDEO2ROBOT_HUMAN_SUFFIX)
    sample_id = sample_id.removesuffix(VIDEO2ROBOT_SKELETON_SUFFIX)
    # Temporal-task ground-truth exports retain a task marker that is absent
    # from the corresponding model rollout directory.
    sample_id = sample_id.removesuffix(TEMPORAL_COMPLETION_GROUNDTRUTH_SUFFIX)
    sample_id = sample_id.removesuffix(TEMPORAL_RETROSPECTION_GROUNDTRUTH_SUFFIX)
    sample_id = sample_id.removesuffix(TEMPORAL_INTERPOLATION_GROUNDTRUTH_SUFFIX)
    # Spatial-completion rollouts append the omitted body region. It is an
    # output-format marker, not part of the source motion identifier.
    for suffix in SPATIAL_COMPLETION_PREDICTION_SUFFIXES:
        sample_id = sample_id.removesuffix(suffix)
    # UH-1 writes ``<source_id>_audio_continuous`` (or ``_audio_pred``).
    # Strip the rollout-state marker first, then the modality marker, so all
    # UH-1 output layouts resolve to the source ID used by ground truth.
    sample_id = sample_id.removesuffix(UH1_CONTINUOUS_SUFFIX)
    sample_id = sample_id.removesuffix(UH1_PREDICTION_SUFFIX)
    sample_id = sample_id.removesuffix(AUDIOMOTION_SUFFIX)
    return sample_id


def sample_id_from_clip_directory(clip: Path, root: Path) -> str:
    """Return the source sample ID for a directory containing motion CSVs.

    Most exporters write CSV files directly under ``<root>/<sample_id>``.
    BFM-ZERO writes ``<root>/<sample_id>/sonic`` instead.  There, ``sonic`` is
    a fixed format directory, not an identifier; using it would collapse all
    BFM-ZERO predictions into one sample.
    """
    if clip.name == BFM_ZERO_LEAF_DIRECTORY and clip.parent != root:
        return normalize_sample_id(clip.parent.name)
    return normalize_sample_id(clip.name)


def _iter_joint_csv_files(root: Path) -> Iterator[Path]:
    """Yield motion CSVs while following directory symlinks safely.

    Level-2 evaluation materializes matched Level-1 clips as directory
    symlinks. ``Path.rglob`` deliberately does not recurse through such
    links, so use ``os.walk(..., followlinks=True)`` and deduplicate by the
    resolved directory inode to retain recursive discovery without allowing
    symlink cycles to recurse indefinitely.
    """
    visited_directories: set[tuple[int, int]] = set()
    for directory_name, child_directories, filenames in os.walk(root, followlinks=True):
        directory = Path(directory_name)
        try:
            stat = directory.stat()
        except OSError:
            child_directories[:] = []
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in visited_directories:
            child_directories[:] = []
            continue
        visited_directories.add(identity)
        # ``os.walk`` has already classified entries in ``filenames``.  Reusing
        # that directory listing avoids three additional metadata round-trips
        # per clip on network-backed Level-2 materialized symlink views.
        if all(name in filenames for name in REQUIRED_CSV):
            yield directory / "joint_pos.csv"


def index_motion_root(root: Path) -> MotionIndex:
    """Recursively index directories containing the three required CSV files.

    Both flat ``root/<sample_id>`` and sharded
    ``root/<shard>/<sample_id>`` layouts are accepted. If extracted data exists
    in both layouts, the shortest relative path wins deterministically.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"motion root does not exist: {root}")
    candidates: dict[str, list[Path]] = {}
    for joint_csv in tqdm(
        _iter_joint_csv_files(root),
        desc=f"Indexing {root.name}",
        unit="clip",
        dynamic_ncols=True,
    ):
        clip = joint_csv.parent
        candidates.setdefault(sample_id_from_clip_directory(clip, root), []).append(clip)
    samples: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for sample_id, paths in candidates.items():
        ordered = sorted(set(paths), key=lambda path: (len(path.relative_to(root).parts), str(path)))
        samples[sample_id] = ordered[0]
        if len(ordered) > 1:
            duplicates[sample_id] = ordered
    if not samples:
        raise RuntimeError(f"no valid motion CSV directories found under {root}")
    return MotionIndex(root=root, samples=samples, duplicates=duplicates)


def _load_csv(path: Path, valid_widths: tuple[int, ...]) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    if value.ndim != 2 or value.shape[1] not in valid_widths:
        raise ValueError(f"unexpected CSV shape for {path}: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite values found in {path}")
    return value


def _continuous_unit_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero-length root quaternion")
    quaternion /= norm
    for frame in range(1, len(quaternion)):
        if float(np.dot(quaternion[frame - 1], quaternion[frame])) < 0.0:
            quaternion[frame] *= -1.0
    return quaternion


def load_qpos_36(clip: Path) -> np.ndarray:
    """Load ``[root xyz, root quaternion wxyz, 29 joint positions]``."""
    joint = _load_csv(clip / "joint_pos.csv", (29,))
    body_pos = _load_csv(clip / "body_pos.csv", tuple(range(3, 121, 3)))[:, :3]
    body_quat = _load_csv(clip / "body_quat.csv", tuple(range(4, 161, 4)))[:, :4]
    frames = min(len(joint), len(body_pos), len(body_quat))
    if frames < 2:
        raise ValueError(f"fewer than two common frames in {clip}")
    root_quat = _continuous_unit_quaternion(body_quat[:frames])
    return np.concatenate((body_pos[:frames], root_quat, joint[:frames]), axis=1).astype(np.float32)
