#!/usr/bin/env python3
"""Aggregate the eleven video2robot Video Level-2 branches into seven scores."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = {}
    for family in ("Speed", "Amplitude", "Direction", "BodyRestrain", "Order", "Times", "Trajectory"):
        branches = ([f"{family}_HumanVideo", f"{family}_SkelVideo"]
                    if family in {"Speed", "Amplitude", "Direction", "BodyRestrain"}
                    else [f"{family}_Video"])
        values = {}
        for branch in branches:
            path = Path("results/level2") / branch / "video2robot" / "level2_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            value = summary.get("bs_level2")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{path}: missing finite numeric bs_level2")
            values[branch] = {"bs_level2": value, "summary": str(path)}
        scores[family] = {
            "bs_level2": sum(item["bs_level2"] for item in values.values()) / len(values),
            "branches": values,
        }
    payload = {
        "schema": "robosteer.level2.video_seven.v1",
        "model": "video2robot",
        "modality": "Video",
        "aggregation": "arithmetic mean of HumanVideo and SkelVideo BS_2 for paired families",
        "scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
