#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ONNX_PATH="${1:-${ROOT_DIR}/deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.onnx}"
CONFIG_OM_PATH="${2:-${ROOT_DIR}/deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
INPUT_SHAPE="${INPUT_SHAPE:-input:1,3,224,224}"
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}"

mkdir -p "$(dirname "${CONFIG_OM_PATH}")"

if ! command -v atc >/dev/null 2>&1; then
  for env_file in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    if [ -f "${env_file}" ]; then
      # shellcheck disable=SC1090
      source "${env_file}"
      break
    fi
  done
fi

if ! command -v atc >/dev/null 2>&1; then
  echo "ERROR: atc not found. Please source Ascend Toolkit set_env.sh first." >&2
  exit 1
fi

if [ ! -f "${CONFIG_ONNX_PATH}" ]; then
  echo "ERROR: ONNX model not found: ${CONFIG_ONNX_PATH}" >&2
  exit 1
fi

atc \
  --model="${CONFIG_ONNX_PATH}" \
  --framework=5 \
  --output="${CONFIG_OM_PATH%.om}" \
  --input_shape="${INPUT_SHAPE}" \
  --soc_version="${SOC_VERSION}" \
  --precision_mode="${PRECISION_MODE}" \
  --log=info
