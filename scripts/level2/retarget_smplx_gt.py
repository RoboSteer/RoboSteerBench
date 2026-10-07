#!/usr/bin/env python3
"""Retarget Level-2 SMPL-X pickle ground truth to evaluator-compatible G1 CSV.

This script is intentionally executed with the dedicated GMR environment.  It
accepts a JSONL manifest with ``sample_id`` and ``source`` fields and writes one
minimal G1 clip directory per unique sample.  Existing complete clips are
resumed without recomputation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
from typing import Any

import numpy as np


REQUIRED_CSV = ("body_pos.csv", "body_quat.csv", "joint_pos.csv")
_GMR_ROOT: Path | None = None
_BODY_MODEL: Any = None
_TORCH: Any = None
_SMPLX_FRAMES: Any = None
_GMR_CLASS: Any = None
_MUJOCO: Any = None
_ROBOT_XML: Path | None = None


def _complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in REQUIRED_CSV)


def _initialize_worker(gmr_root: str) -> None:
    global _GMR_ROOT, _BODY_MODEL, _TORCH, _SMPLX_FRAMES, _GMR_CLASS, _MUJOCO, _ROBOT_XML
    _GMR_ROOT = Path(gmr_root).resolve()
    sys.path.insert(0, str(_GMR_ROOT))
    import mujoco
    import smplx
    import torch
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting import ROBOT_XML_DICT
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

    torch.set_num_threads(1)
    _TORCH = torch
    _SMPLX_FRAMES = get_smplx_data_offline_fast
    _GMR_CLASS = GeneralMotionRetargeting
    _MUJOCO = mujoco
    _ROBOT_XML = Path(ROBOT_XML_DICT["unitree_g1"])
    _BODY_MODEL = smplx.create(
        _GMR_ROOT / "assets" / "body_models",
        "smplx",
        gender="neutral",
        use_pca=False,
        ext="pkl",
    )


def _convert_root_y_up_to_z_up(
    root_pos: np.ndarray, root_rot_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as Rotation

    matrix = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rotation = Rotation.from_matrix(matrix)
    root_pos = root_pos @ matrix.T
    root_xyzw = Rotation.from_quat(root_rot_wxyz[:, [1, 2, 3, 0]])
    root_xyzw = (rotation * root_xyzw).as_quat()
    return root_pos, root_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


def _ground_align(
    root_pos: np.ndarray, root_rot_wxyz: np.ndarray, dof_pos: np.ndarray
) -> np.ndarray:
    assert _MUJOCO is not None and _ROBOT_XML is not None
    model = _MUJOCO.MjModel.from_xml_path(str(_ROBOT_XML))
    data = _MUJOCO.MjData(model)
    body_ids = [
        model.body(name).id
        for name in (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_toe_link",
            "right_toe_link",
        )
    ]
    minimum = np.inf
    for position, quaternion, joints in zip(root_pos, root_rot_wxyz, dof_pos):
        data.qpos[:3] = position
        data.qpos[3:7] = quaternion
        data.qpos[7:] = joints
        _MUJOCO.mj_forward(model, data)
        minimum = min(minimum, *(float(data.xpos[index][2]) for index in body_ids))
    result = root_pos.copy()
    result[:, 2] += 0.005 - minimum
    return result


def _write_csv(path: Path, values: np.ndarray, header: str) -> None:
    np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.8f")


def _retarget_one(item: tuple[str, str, str]) -> dict[str, Any]:
    sample_id, source_string, destination_string = item
    source = Path(source_string)
    destination = Path(destination_string)
    if _complete(destination):
        return {"sample_id": sample_id, "status": "cached", "frames": None}
    started = time.monotonic()
    with source.open("rb") as handle:
        raw = pickle.load(handle)
    poses = np.asarray(raw["poses"], dtype=np.float32)
    translation = np.asarray(raw["trans"], dtype=np.float32)
    fps = float(raw["fps"])
    if poses.ndim != 2 or poses.shape[1] < 66 or translation.shape != (len(poses), 3):
        raise ValueError(
            f"{source}: expected poses [T,>=66] and trans [T,3], got "
            f"{poses.shape} and {translation.shape}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(translation).all() or fps <= 0:
        raise ValueError(f"{source}: invalid non-finite motion data or FPS")
    assert _TORCH is not None and _BODY_MODEL is not None and _SMPLX_FRAMES is not None
    frames = len(poses)
    smplx_data = {
        "pose_body": poses[:, 3:66],
        "root_orient": poses[:, :3],
        "trans": translation,
        "betas": np.zeros(16, dtype=np.float32),
        "mocap_frame_rate": np.asarray(fps, dtype=np.float32),
    }
    with _TORCH.no_grad():
        output = _BODY_MODEL(
            betas=_TORCH.zeros((1, 16), dtype=_TORCH.float32),
            global_orient=_TORCH.from_numpy(smplx_data["root_orient"]),
            body_pose=_TORCH.from_numpy(smplx_data["pose_body"]),
            transl=_TORCH.from_numpy(smplx_data["trans"]),
            left_hand_pose=_TORCH.zeros((frames, 45)),
            right_hand_pose=_TORCH.zeros((frames, 45)),
            jaw_pose=_TORCH.zeros((frames, 3)),
            leye_pose=_TORCH.zeros((frames, 3)),
            reye_pose=_TORCH.zeros((frames, 3)),
            return_full_pose=True,
        )
    human_frames, aligned_fps = _SMPLX_FRAMES(
        smplx_data, _BODY_MODEL, output, tgt_fps=30
    )
    # GMR prints the full robot topology on every construction; suppress that
    # deterministic diagnostic so large batch logs remain usable.
    with contextlib.redirect_stdout(io.StringIO()):
        retargeter = _GMR_CLASS(
            actual_human_height=1.66,
            src_human="smplx",
            tgt_robot="unitree_g1",
        )
    qpos = np.asarray([retargeter.retarget(frame) for frame in human_frames], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError(f"{source}: invalid GMR output {qpos.shape}")
    root_pos, root_quat = _convert_root_y_up_to_z_up(qpos[:, :3], qpos[:, 3:7])
    joints = qpos[:, 7:]
    root_pos = _ground_align(root_pos, root_quat, joints)

    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    _write_csv(temporary / "body_pos.csv", root_pos, "body_0_x,body_0_y,body_0_z")
    _write_csv(temporary / "body_quat.csv", root_quat, "body_0_w,body_0_x,body_0_y,body_0_z")
    _write_csv(
        temporary / "joint_pos.csv",
        joints,
        ",".join(f"joint_{index}" for index in range(joints.shape[1])),
    )
    (temporary / "metadata.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_fps": fps,
                "fps": float(aligned_fps),
                "frames": len(qpos),
                "retargeter": "GMR unitree_g1",
                "body_shape": "SMPL-X neutral, zero betas (benchmark GT omits shape)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _complete(destination):
            shutil.rmtree(temporary)
            return {"sample_id": sample_id, "status": "cached", "frames": len(qpos)}
        shutil.rmtree(destination)
    temporary.rename(destination)
    return {
        "sample_id": sample_id,
        "status": "converted",
        "frames": len(qpos),
        "seconds": time.monotonic() - started,
    }


def _read_manifest(path: Path, output: Path) -> list[tuple[str, str, str]]:
    items: dict[str, tuple[str, str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        source = Path(str(row["source"])).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: GT does not exist: {source}")
        item = (sample_id, str(source), str((output / sample_id).resolve()))
        previous = items.get(sample_id)
        if previous is not None and previous[1] != item[1]:
            raise ValueError(f"{path}:{line_number}: conflicting sources for {sample_id}")
        items[sample_id] = item
    if not items:
        raise RuntimeError(f"empty GT manifest: {path}")
    return [items[key] for key in sorted(items)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = _read_manifest(args.manifest.expanduser().resolve(), output)
    cached = sum(_complete(Path(item[2])) for item in items)
    pending = [item for item in items if not _complete(Path(item[2]))]
    completed = cached
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    print(f"Level2 GT retarget: {len(items)} total, {cached} cached, {len(pending)} pending", flush=True)
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            initializer=_initialize_worker,
            initargs=(str(args.gmr_root.expanduser().resolve()),),
        ) as executor:
            future_to_item = {executor.submit(_retarget_one, item): item for item in pending}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    future.result()
                    completed += 1
                except Exception as exc:
                    failures.append({"sample_id": item[0], "source": item[1], "error": repr(exc)})
                if completed % 25 == 0 or completed + len(failures) == len(items):
                    elapsed = time.monotonic() - started
                    print(
                        f"Level2 GT retarget: {completed}/{len(items)} complete, "
                        f"{len(failures)} failed, {elapsed:.1f}s",
                        flush=True,
                    )
                if args.progress:
                    args.progress.parent.mkdir(parents=True, exist_ok=True)
                    args.progress.write_text(
                        json.dumps(
                            {
                                "total": len(items),
                                "complete": completed,
                                "cached_at_start": cached,
                                "failed": failures,
                                "elapsed_seconds": time.monotonic() - started,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
    if failures:
        raise RuntimeError(f"failed to retarget {len(failures)} Level2 GT motions")
    print(f"Level2 GT retarget complete: {completed}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
