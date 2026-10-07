#!/usr/bin/env python3
"""Build the uploadable RoboSteer evaluator bundle in models/.

The source code lives in evaluator_code/ beside models/. Large weights are
hard-linked when possible; the Hub upload still reads full files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
OMG_SHA256 = "e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40"
CHECKPOINTS = {
    "audio_motion_evaluator": "audio_motion_evaluator.pt",
    "rhythm_motion_evaluator": "rhythm_motion_evaluator.pt",
    "trajectory_motion_evaluator": "rotation_motion_evaluator.pt",
    "text_motion_evaluator": "text_motion_evaluator.pt",
    "video_motion_evaluator_human": "human_video_motion_evaluator.pt",
    "video_motion_evaluator_skel": "skeleton_video_motion_evaluator.pt",
}


def place(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_snapshot(snapshot: Path, destination: Path, names: list[str]) -> None:
    for name in names:
        place(snapshot / name, destination / name)


def registry_payload(selected: str | None = None) -> dict:
    source = json.loads((PROJECT / "evaluator_code" / "registry.json").read_text(encoding="utf-8"))
    source["motion_encoder"]["checkpoint"] = "motion_encoder/weights.pt"
    source["motion_encoder"]["architecture"] = (
        "../../evaluator_code/motion_encoder/model.py" if selected else "../evaluator_code/motion_encoder/model.py"
    )
    for key, config in source["cross_modal_evaluators"].items():
        name = config["root"]
        prefix = "" if selected else f"{name}/"
        config["root"] = f"{'../..' if selected else '..'}/evaluator_code/{name}"
        config["checkpoint"] = f"{prefix}checkpoints/{CHECKPOINTS[name]}"
        config["omg_root"] = f"{prefix}omg"
        config["omg_checkpoint"] = f"{prefix}omg/models/evaluator/step_004000.pt"
        if key == "audio_motion":
            config["whisper_model_path"] = f"{prefix}whisper-small"
        elif key == "rhythm_motion":
            config["ast_model_path"] = f"{prefix}ast-audioset"
            config.pop("hf_home", None)
        elif key == "text_motion":
            config["text_model_path"] = f"{prefix}t5-base"
        elif key in ("video_motion_human", "video_motion_skel"):
            config["xclip_model_path"] = f"{prefix}xclip-base-patch32"
            config["xclip_cache"] = f"{prefix}xclip-base-patch32"
    if selected is not None:
        source["cross_modal_evaluators"] = {
            selected: source["cross_modal_evaluators"][selected]
        }
    return source


def motion_only_registry() -> dict:
    source = registry_payload()
    source["motion_encoder"]["checkpoint"] = "weights.pt"
    source["motion_encoder"]["architecture"] = "../../evaluator_code/motion_encoder/model.py"
    source["cross_modal_evaluators"] = {}
    return source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def omg_provenance(source: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain", "--", "src/omg", "assets/robots/g1")
    return {
        "upstream": "https://github.com/Tsinghua-MARS-Lab/OMG",
        "source_commit": git("rev-parse", "HEAD"),
        "local_changes_in_bundled_files": status.splitlines() if status else [],
        "checkpoint_sha256": OMG_SHA256,
    }


def stage_omg(source_root: Path, destination: Path, *, checkpoint: bool) -> None:
    place(source_root / "LICENSE", destination / "LICENSE")
    asset_root = source_root / "assets" / "robots" / "g1"
    for name in ("g1_kinematics.json", "g1_29dof.urdf"):
        place(asset_root / name, destination / "assets" / "robots" / "g1" / name)
    for source in (source_root / "src" / "omg").rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
            place(source, destination / "src" / "omg" / source.relative_to(source_root / "src" / "omg"))
    if checkpoint:
        place(source_root / "models" / "evaluator" / "step_004000.pt",
              destination / "models" / "evaluator" / "step_004000.pt")
    (destination / "SOURCE.json").write_text(
        json.dumps(omg_provenance(source_root), indent=2) + "\n", encoding="utf-8"
    )


def stage_backbone(name: str, destination: Path) -> None:
    if name == "text_motion_evaluator":
        stage_snapshot(PROJECT / "models" / name, destination / "t5-base",
                       ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"])
    elif name == "audio_motion_evaluator":
        stage_snapshot(
            PROJECT / "models" / name / "model_cache" / "models--openai--whisper-small" /
            "snapshots" / "973afd24965f72e36ca33b3055d56a652f456b4d",
            destination / "whisper-small",
            ["config.json", "model.safetensors", "preprocessor_config.json"],
        )
    elif name == "rhythm_motion_evaluator":
        stage_snapshot(
            PROJECT / "models" / name / "hf_cache" / "hub" /
            "models--MIT--ast-finetuned-audioset-10-10-0.4593" / "snapshots" /
            "f826b80d28226b62986cc218e5cec390b1096902",
            destination / "ast-audioset",
            ["config.json", "model.safetensors", "preprocessor_config.json"],
        )
    elif name in ("video_motion_evaluator_human", "video_motion_evaluator_skel"):
        stage_snapshot(
            PROJECT / "models" / "video_motion_evaluator_human" / "model_cache" /
            "models--microsoft--xclip-base-patch32" / "snapshots" /
            "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
            destination / "xclip-base-patch32",
            ["config.json", "model.safetensors", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt",
             "preprocessor_config.json", "tokenizer.json"],
        )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build(destination: Path, omg_source: Path) -> None:
    if destination.resolve() == (PROJECT / "evaluator_code").resolve():
        raise ValueError("destination must not be the evaluator source directory")
    destination.mkdir(parents=True, exist_ok=True)
    omg_checkpoint = omg_source / "models" / "evaluator" / "step_004000.pt"
    if sha256(omg_checkpoint) != OMG_SHA256:
        raise ValueError("OMG checkpoint does not match the evaluator training revision")

    motion_root = destination / "motion_encoder"
    place(PROJECT / "models" / "motion_encoder" / "weights.pt", motion_root / "weights.pt")
    stage_omg(omg_source, motion_root / "omg", checkpoint=False)
    write_json(motion_root / "registry.json", motion_only_registry())
    (motion_root / "README.md").write_text(
        "# Motion encoder\n\nWeights and OMG code/assets for FID and Diversity.\n", encoding="utf-8"
    )

    source_registry = json.loads((PROJECT / "evaluator_code" / "registry.json").read_text(encoding="utf-8"))
    for key, config in source_registry["cross_modal_evaluators"].items():
        name = config["root"]
        component = destination / name
        place(PROJECT / "models" / name / "checkpoints" / CHECKPOINTS[name],
              component / "checkpoints" / CHECKPOINTS[name])
        place(PROJECT / "models" / "motion_encoder" / "weights.pt",
              component / "motion_encoder" / "weights.pt")
        stage_omg(omg_source, component / "omg", checkpoint=True)
        stage_backbone(name, component)
        write_json(component / "registry.json", registry_payload(key))
        (component / "README.md").write_text(
            f"# {name}\n\nSelf-contained weights for this RoboSteer evaluator. "
            "Use with the matching GitHub source code and "
            f"`--models models/{name}/registry.json`.\n",
            encoding="utf-8",
        )

    write_json(destination / "registry.json", registry_payload())
    (destination / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    verify(destination)
    # Remove files created by the previous shared-folder layout only after the
    # replacement has passed verification.
    for old_name in ("backbones", "evaluators", "omg"):
        old = destination / old_name
        if old.is_dir():
            shutil.rmtree(old)


def verify_registry(root: Path, *, omg_checkpoint: bool) -> None:
    registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    required = [root / registry["motion_encoder"]["checkpoint"],
                root / registry["motion_encoder"]["architecture"]]
    for config in registry["cross_modal_evaluators"].values():
        for key in ("root", "checkpoint", "omg_root", "omg_checkpoint"):
            required.append(root / config[key])
        for key in ("whisper_model_path", "ast_model_path", "text_model_path", "xclip_model_path"):
            if key in config:
                required.append(root / config[key] / "model.safetensors")
    required.extend([
        root / "omg" / "src" / "omg" / "__init__.py",
        root / "omg" / "assets" / "robots" / "g1" / "g1_kinematics.json",
        root / "omg" / "assets" / "robots" / "g1" / "g1_29dof.urdf",
        root / "omg" / "SOURCE.json",
        root / "README.md",
    ])
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("incomplete models component: " + ", ".join(map(str, missing)))
    checkpoint = root / "omg" / "models" / "evaluator" / "step_004000.pt"
    if omg_checkpoint and sha256(checkpoint) != OMG_SHA256:
        raise ValueError(f"OMG checkpoint SHA256 mismatch: {checkpoint}")


def verify(destination: Path, component: str | None = None) -> None:
    if component is not None:
        if component not in {*CHECKPOINTS, "motion_encoder"}:
            raise ValueError(f"unknown component: {component}")
        root = destination / component
        verify_registry(root, omg_checkpoint=component != "motion_encoder")
        print(f"Verified {root}")
        return
    available = [name for name in (*CHECKPOINTS, "motion_encoder")
                 if (destination / name / "registry.json").is_file()]
    if len(available) == 1 and not (destination / "registry.json").is_file():
        verify(destination, available[0])
        return
    if len(available) != len(CHECKPOINTS) + 1:
        raise ValueError("partial bundle: pass --component NAME to verify one evaluator")
    verify_registry(destination / "motion_encoder", omg_checkpoint=False)
    for name in CHECKPOINTS:
        verify_registry(destination / name, omg_checkpoint=True)
    global_registry = json.loads((destination / "registry.json").read_text(encoding="utf-8"))
    for key, config in global_registry["cross_modal_evaluators"].items():
        name = config["root"].split("/")[-1]
        if not (destination / config["checkpoint"]).is_file() or name not in CHECKPOINTS:
            raise FileNotFoundError(f"global registry has invalid {key} checkpoint")
    print(f"Verified {destination}")


MODEL_CARD = """---
tags:
- robosteer
- evaluation
---

# RoboSteer evaluator weights

This repository groups RoboSteer evaluator weights by purpose. Each evaluator
folder includes its checkpoint, required backbone, OMG dependency, motion
encoder, and a local registry. Download only the folder you need. The separate
Order/Times Qwen3-VL model is not included.

## Install

Download one evaluator into `models/` next to the RoboSteer GitHub
repository's `evaluator_code/` directory, for example:

```bash
hf download Leontc/RoboSteer-evaluator \
  --include 'text_motion_evaluator/**' --local-dir models
```

The evaluator implementation lives in the GitHub repository. When only one
evaluator folder is downloaded, evaluation commands select its registry
automatically. You can also pass
`--models models/text_motion_evaluator/registry.json`. For all evaluators,
download the whole repository into `models/`; its root `registry.json`
selects every folder. Keep `evaluator_code/` and `models/` as siblings and install
the GitHub project's Python requirements first.

## Contents

- `text_motion_evaluator/`: text checkpoint and T5-base.
- `audio_motion_evaluator/`: audio checkpoint and Whisper-small.
- `rhythm_motion_evaluator/`: rhythm checkpoint and AST AudioSet.
- `trajectory_motion_evaluator/`: trajectory checkpoint.
- `video_motion_evaluator_human/` and `video_motion_evaluator_skel/`:
  video checkpoints and XCLIP.
- `motion_encoder/`: standalone FID/Diversity weights and preprocessing assets.

Each evaluator folder also includes `motion_encoder/weights.pt`, an `omg/`
directory, and its own `registry.json`, so it works when downloaded alone.
`omg/SOURCE.json` records the OMG source commit and local changes.

All model files are ordinary files; no Hugging Face cache symlinks or duplicate
AST PyTorch/safetensors copies are required. The exact OMG checkpoint SHA256
is `e346790c5ce33c42840aa19f7fcc2891a4cab27e0d92e23bfaff7d48cb339c40`.

## Upstream models and licenses

- [OMG](https://github.com/Tsinghua-MARS-Lab/OMG) source: MIT. Each folder includes its license; verify the checkpoint's redistribution terms before public release.
- [google-t5/t5-base](https://huggingface.co/google-t5/t5-base): Apache 2.0.
- [openai/whisper-small](https://huggingface.co/openai/whisper-small): Apache 2.0.
- [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593): BSD 3-Clause.
- [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32): MIT.

Pinned upstream revisions: T5 `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`,
Whisper `973afd24965f72e36ca33b3055d56a652f456b4d`, AST
`f826b80d28226b62986cc218e5cec390b1096902`, and XCLIP
`a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.

The RoboSteer-specific checkpoint license should be stated here before public
publication. These upstream licenses remain applicable to their respective
files; a single blanket license for the bundle would be misleading.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "models")
    parser.add_argument(
        "--omg-source", type=Path,
        default=Path(os.environ.get("ROBOOSTEER_OMG_ROOT", PROJECT / "third_party" / "OMG")),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--component", choices=[*CHECKPOINTS, "motion_encoder"])
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output.resolve(), args.component)
    else:
        build(args.output.resolve(), args.omg_source.resolve())
