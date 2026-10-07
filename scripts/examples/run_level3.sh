#!/usr/bin/env bash
set -euo pipefail

# One-click Level-3 example. A non-image interleave item is evaluated as its
# matching Level-2 generation subtask (IR_1 = IR_2 = 1); image boundaries are
# ignored and the remaining BS_level2 values are duration-weighted.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-python3}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the benchmark dataset root}"
TEXT_PREDICTION_ROOT="${TEXT_PREDICTION_ROOT:?Set TEXT_PREDICTION_ROOT to Level 3 text predictions}"
AUDIO_PREDICTION_ROOT="${AUDIO_PREDICTION_ROOT:?Set AUDIO_PREDICTION_ROOT to Level 3 audio predictions}"
VIDEO_PREDICTION_ROOT="${VIDEO_PREDICTION_ROOT:?Set VIDEO_PREDICTION_ROOT to Level 3 video predictions}"
TASK_ROOT="${TASK_ROOT:-${DATASET_ROOT}/Tasks/Level3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/level3/GEM-GMR}"
MM_GPUS="${MM_GPUS:-0}"

exec "${EVALUATION_PYTHON}" "${SCRIPT_DIR}/../level3/evaluate.py" \
  --text-prediction "${TEXT_PREDICTION_ROOT}" \
  --audio-prediction "${AUDIO_PREDICTION_ROOT}" \
  --video-prediction "${VIDEO_PREDICTION_ROOT}" \
  --task-root "${TASK_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --metrics FID,MM-Distance,BG,BS_3 \
  --mm-gpus "${MM_GPUS}" \
  "$@"
