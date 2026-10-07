#!/usr/bin/env bash
set -euo pipefail

# One-click Level-1 example: GEM-GMR text-to-motion generation.
# Override any variable on the command line, e.g.
# PREDICTION_ROOT=/path/to/your/predictions bash scripts/examples/run_level1.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-python3}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the benchmark dataset root}"
PREDICTION_ROOT="${PREDICTION_ROOT:?Set PREDICTION_ROOT to the model predictions}"
MOTION_GROUNDTRUTH_ROOT="${MOTION_GROUNDTRUTH_ROOT:-${DATASET_ROOT}/Data/Shared/Motion}"
CONDITION_GROUNDTRUTH_ROOT="${CONDITION_GROUNDTRUTH_ROOT:-${DATASET_ROOT}/Tasks/Level1/full_conditioning_reproduction/text_to_motion_generation}"
TASK_ROOT="${TASK_ROOT:-${CONDITION_GROUNDTRUTH_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/level1/Text_GEM}"
MM_GPUS="${MM_GPUS:-0}"

exec "${EVALUATION_PYTHON}" "${SCRIPT_DIR}/../level1/evaluate.py" \
  --prediction "${PREDICTION_ROOT}" \
  --motion-groundtruth "${MOTION_GROUNDTRUTH_ROOT}" \
  --condition-groundtruth "${CONDITION_GROUNDTRUTH_ROOT}" \
  --task-root "${TASK_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --modality text \
  --output "${OUTPUT_ROOT}" \
  --metrics FID,MM-Distance,BG,BS_1 \
  --mm-encoder text_motion \
  --mm-gpus "${MM_GPUS}" \
  "$@"
