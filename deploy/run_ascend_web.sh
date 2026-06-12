#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-${ROOT_DIR}/deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om}"
VIDEO_SOURCE="${2:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${ROOT_DIR}/${MODEL_PATH}"
fi
if [[ ! "${VIDEO_SOURCE}" =~ ^[0-9]+$ && "${VIDEO_SOURCE}" != /* ]]; then
  VIDEO_SOURCE="${ROOT_DIR}/${VIDEO_SOURCE}"
fi

if ! "${PYTHON_BIN}" -c "import acl" >/dev/null 2>&1; then
  for env_file in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/nnrt/set_env.sh \
    /usr/local/Ascend/nnrt/latest/set_env.sh; do
    if [ -f "${env_file}" ]; then
      # shellcheck disable=SC1090
      source "${env_file}"
      break
    fi
  done
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: OM model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import acl, cv2, numpy, PIL, yaml" >/dev/null 2>&1; then
  echo "ERROR: Python dependencies are incomplete." >&2
  echo "Source the CANN environment and install deploy/requirements_board.txt." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" scripts/ascend_web_demo.py \
  --config configs/config.yaml \
  --model "${MODEL_PATH}" \
  --source "${VIDEO_SOURCE}" \
  --device-id "${DEVICE_ID}" \
  --host "${HOST}" \
  --port "${PORT}"
