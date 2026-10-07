"""Compute a post-hoc BG score from completed FID and MM-Distance results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.core_metric.BehaviorGenerationScore import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_SCALE,
    compute_behavior_generation_score,
)


def _read_metric(path: Path, expected_name: str) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"required metric file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    name = payload.get("name")
    if name is not None and name != expected_name:
        raise ValueError(f"{path}: expected metric name {expected_name!r}, got {name!r}")
    if "value" not in payload:
        raise ValueError(f"{path}: missing numeric 'value' field")

    value = float(payload["value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: metric value must be finite and non-negative, got {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BG from metrics/fid.json and metrics/mm_distance.json."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing fid.json and mm_distance.json.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"BG output scale (default: {DEFAULT_SCALE}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"FID coefficient (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"MM-Distance coefficient (default: {DEFAULT_BETA}).",
    )
    parser.add_argument(
        "--output-name",
        default="bg.json",
        help="Output JSON filename written inside --metrics-dir (default: bg.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir.expanduser().resolve()
    if not metrics_dir.is_dir():
        raise NotADirectoryError(f"metrics directory does not exist: {metrics_dir}")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a filename, not a path")

    fid_path = metrics_dir / "fid.json"
    mm_distance_path = metrics_dir / "mm_distance.json"
    fid = _read_metric(fid_path, "FID")
    mm_distance = _read_metric(mm_distance_path, "MM-Distance")
    value = compute_behavior_generation_score(
        fid,
        mm_distance,
        scale=args.scale,
        alpha=args.alpha,
        beta=args.beta,
    )

    output_path = metrics_dir / args.output_name
    result = {
        "name": "BG",
        "value": value,
        "direction": "higher_is_better",
        "formula": "scale * exp(-alpha * FID - beta * MM-Distance)",
        "fid": fid,
        "mm_distance": mm_distance,
        "scale": args.scale,
        "alpha": args.alpha,
        "beta": args.beta,
        "source_metrics": {
            "fid": str(fid_path),
            "mm_distance": str(mm_distance_path),
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({**result, "path": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
