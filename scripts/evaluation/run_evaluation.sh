#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_PYTHON="${EVALUATION_PYTHON:-python3}"

exec "${EVALUATION_PYTHON}" "${SCRIPT_DIR}/evaluate.py" "$@"
