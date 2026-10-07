#!/usr/bin/env bash
set -euo pipefail

# One-click Level-2 example: video2robot-GMR, skeleton-video Speed steering.
# BG is computed only from the matching IDs in the Level-1 base prediction,
# shared GT, and original Level-1 skeleton-video condition.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-python3}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the benchmark dataset root}"
LEVEL2_PREDICTION_ROOT="${LEVEL2_PREDICTION_ROOT:?Set LEVEL2_PREDICTION_ROOT to Level 2 predictions}"
BASE_PREDICTION_ROOT="${BASE_PREDICTION_ROOT:?Set BASE_PREDICTION_ROOT to Level 1 predictions}"
BASE_MOTION_GROUNDTRUTH_ROOT="${BASE_MOTION_GROUNDTRUTH_ROOT:-${DATASET_ROOT}/Data/Shared/Motion}"
BASE_CONDITION_GROUNDTRUTH_ROOT="${BASE_CONDITION_GROUNDTRUTH_ROOT:-${DATASET_ROOT}/Tasks/Level1/full_conditioning_reproduction/video_to_motion_imitation}"
LEVEL2_TASK_ROOT="${LEVEL2_TASK_ROOT:-${DATASET_ROOT}/Tasks/Level2/Speed/skeleton_video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/level2/Speed_SkelVideo/video2robot}"
MM_GPUS="${MM_GPUS:-0}"

exec "${EVALUATION_PYTHON}" "${SCRIPT_DIR}/../level2/evaluate.py" \
  --prediction "${LEVEL2_PREDICTION_ROOT}" \
  --level2-task-root "${LEVEL2_TASK_ROOT}" \
  --base-prediction "${BASE_PREDICTION_ROOT}" \
  --base-motion-groundtruth "${BASE_MOTION_GROUNDTRUTH_ROOT}" \
  --base-condition-groundtruth "${BASE_CONDITION_GROUNDTRUTH_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --modality skeleton_video \
  --output "${OUTPUT_ROOT}" \
  --metrics FID,MM-Distance,BG,BS_2 \
  --mm-encoder video_motion_skel \
  --mm-gpus "${MM_GPUS}" \
  "$@"
