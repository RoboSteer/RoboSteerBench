#!/usr/bin/env python3
"""Run conventional metrics from explicit prediction, GT, and condition roots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.conventional.runner import run_conventional
from scripts.evaluation.model_paths import default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--motion-groundtruth", type=Path, required=True)
    parser.add_argument("--instruction-groundtruth", type=Path, required=True)
    parser.add_argument("--task-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mm-encoder", required=True)
    parser.add_argument("--metrics", nargs="+", default=("FID", "MM-Distance", "BG"))
    parser.add_argument("--models", type=Path, default=default_registry(PROJECT_ROOT))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "evaluation" / "default.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    metrics = tuple(name.strip() for group in args.metrics for name in group.split(",") if name.strip())
    run_conventional(
        prediction=args.prediction,
        motion_groundtruth=args.motion_groundtruth,
        instruction_groundtruth=args.instruction_groundtruth,
        task_metadata=args.task_metadata,
        output=args.output,
        mm_encoder=args.mm_encoder,
        metrics=metrics,
        models=args.models,
        config_path=args.config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
