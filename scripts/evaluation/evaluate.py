#!/usr/bin/env python3
"""Evaluate selected metrics from prediction and optional ground-truth roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 使脚本可以直接运行，并能导入 scripts.evaluation 内部模块。
LIUJIAHUI_ROOT = Path(__file__).resolve().parents[2]
if str(LIUJIAHUI_ROOT) not in sys.path:
    sys.path.insert(0, str(LIUJIAHUI_ROOT))

from scripts.evaluation.model_paths import default_registry


def main() -> None:
    # 定义评测命令行参数；路径、指标和模型配置最终会传入评测管线。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, help="generated motion CSV root")
    parser.add_argument("--motion-groundtruth", type=Path, help="reference motion CSV root")
    parser.add_argument(
        "--instruction-groundtruth",
        type=Path,
        help=(
            "instruction root or manifest.json; MM-Distance uses the selected paired "
            "cross-modal evaluator, while BAS expects audio-feature matrices"
        ),
    )
    parser.add_argument("--output", type=Path, help="evaluation output root")
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="metric names separated by spaces and/or commas, for example FID,Diversity",
    )
    parser.add_argument(
        "--mm-encoder",
        help=(
            "registered paired X-motion evaluator used by MM-Distance and Recall@K, "
            "for example audio_motion, text_motion, or video_motion_human"
        ),
    )
    parser.add_argument(
        "--mm-gpus",
        help=(
            "comma-separated GPU indices dedicated to MM pair encoding, for example "
            "0,1,2,3; each device runs one spawn-isolated worker"
        ),
    )
    parser.add_argument(
        "--video-cache-root",
        type=Path,
        help="directory for byte-identical cached condition videos",
    )
    parser.add_argument(
        "--mm-video-decode-workers-per-gpu",
        type=int,
        help="concurrent video cache/decode workers in each MM GPU process",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=default_registry(LIUJIAHUI_ROOT),
        help="encoder registry JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=LIUJIAHUI_ROOT / "configs" / "evaluation" / "default.json",
        help="shared temporal/evaluation protocol JSON",
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=LIUJIAHUI_ROOT / "data" / "metadata.jsonl",
        help=(
            "OMG task JSON directory or duration manifest; metadata.duration is the authoritative "
            "GT timeline and is recorded in provenance"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--list-metrics", action="store_true")
    args = parser.parse_args()

    # 仅展示已注册指标、优化方向及其所需输入，不启动实际评测。
    if args.list_metrics:
        print("FID\tlower_is_better\tprediction_motion,motion_groundtruth")
        print("Diversity\thigher_is_better\tprediction_motion")
        print("ContactSliding\tlower_is_better\tprediction_motion")
        print("MPJPE\tlower_is_better\tprediction_motion,motion_groundtruth")
        print("g-MPJPE\tlower_is_better\tprediction_motion,motion_groundtruth")
        print("E_vel\tlower_is_better\tprediction_motion,motion_groundtruth")
        print("MM-Distance\tlower_is_better\tprediction_motion,instruction_groundtruth,mm_encoder")
        print("BG\thigher_is_better\tprediction_motion,motion_groundtruth,instruction_groundtruth")
        print("BS_level1\thigher_is_better\tprediction_motion,motion_groundtruth,instruction_groundtruth")
        print("BS_level2\thigher_is_better\tprediction_motion,motion_groundtruth,instruction_groundtruth,level2_speed_amplitude_bodyrestrain_direction_or_trajectory_task_json")
        print("R@1\thigher_is_better\tprediction_motion,instruction_groundtruth,mm_encoder")
        print("R@5\thigher_is_better\tprediction_motion,instruction_groundtruth,mm_encoder")
        print("R@10\thigher_is_better\tprediction_motion,instruction_groundtruth,mm_encoder")
        print("BAS-Gen\thigher_is_better\tprediction_motion,motion_groundtruth,instruction_groundtruth(audio_features)")
        print("BAS-Gap\tcloser_to_zero\tprediction_motion,motion_groundtruth,instruction_groundtruth(audio_features)")
        return
    # 检查所有评测共同需要的基础参数，缺失时尽早报错。
    missing = [
        name
        for name, value in (
            ("--prediction", args.prediction),
            ("--output", args.output),
            ("--metrics", args.metrics),
        )
        if value is None
    ]

    if missing:
        parser.error(f"required arguments missing: {', '.join(missing)}")

    # 读取共享评测协议，例如 FPS、窗口长度和批处理大小。
    config_path = args.config.resolve()
    if not config_path.is_file():
        parser.error(f"evaluation config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # 允许 --metrics 同时使用空格或逗号分隔，并统一去除空白项。
    metrics = tuple(
        name.strip()
        for group in args.metrics
        for name in group.split(",")
        if name.strip()
    )
    # Recall@K 固定成同一检索协议下的 R@1、R@5、R@10 三项套件。
    retrieval_metrics = {"r1", "r5", "r10"}
    # 请求其中任意一项时，都展开为三项，并复用相同候选分组。
    if any(_normal(name) in retrieval_metrics for name in metrics):
        metrics = tuple(
            name for name in metrics if _normal(name) not in retrieval_metrics
        ) + ("R@1", "R@5", "R@10")
    # MM-Distance/检索及其 Level-2 派生指标都需要明确的条件模态。
    mm_dependent_metrics = {
        "mmdistance", *retrieval_metrics,
        "bg", "bs1", "bslevel1", "bs2", "bslevel2",
    }
    if any(_normal(name) in mm_dependent_metrics for name in metrics):
        if not args.mm_encoder:
            parser.error(
                "MM-Distance, Recall@K, BG, and BS require --mm-encoder, "
                "for example audio_motion"
            )
    if args.mm_encoder:
        config = dict(config)
        config["mm_encoder"] = args.mm_encoder
    if args.mm_gpus:
        gpu_ids = [item.strip() for item in args.mm_gpus.split(",") if item.strip()]
        if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
            parser.error("--mm-gpus must be a comma-separated list of non-negative GPU indices")
        if len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--mm-gpus must not contain duplicate GPU indices")
        config = dict(config)
        config["mm_gpu_devices"] = [f"cuda:{int(item)}" for item in gpu_ids]
    if args.video_cache_root is not None:
        config = dict(config)
        config["video_local_cache_root"] = str(args.video_cache_root.resolve())
    if args.mm_video_decode_workers_per_gpu is not None:
        if args.mm_video_decode_workers_per_gpu <= 0:
            parser.error("--mm-video-decode-workers-per-gpu must be positive")
        config = dict(config)
        config["mm_video_decode_workers_per_gpu"] = args.mm_video_decode_workers_per_gpu

    # 延迟导入评测管线，避免 --list-metrics 时加载模型相关依赖。
    from scripts.evaluation.pipeline.evaluator import EvaluationRequest, run_evaluation

    # 将命令行参数封装为一次可复现评测请求，并交给统一管线执行。
    request = EvaluationRequest(
        prediction=args.prediction,
        motion_groundtruth=args.motion_groundtruth,
        instruction_groundtruth=args.instruction_groundtruth,
        output=args.output,
        metrics=metrics,
        models=args.models,
        task_metadata=args.task_metadata,
        config_path=config_path,
        device=args.device,
    )
    summary = run_evaluation(request, config)
    # 在终端打印摘要；完整 JSON、指标文件和缓存会写入 --output。
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _normal(name: str) -> str:
    # 将 "R@1"、"r1" 等写法归一化，便于命令行别名匹配。
    return "".join(character.lower() for character in name if character.isalnum())


if __name__ == "__main__":
    main()
