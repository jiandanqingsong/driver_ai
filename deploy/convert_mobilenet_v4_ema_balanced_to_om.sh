#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-${ROOT_DIR}/deploy/models/mobilenet_v4_ema_demo_finetune_balanced_driver_distraction.onnx}"
OM_PATH="${2:-${ROOT_DIR}/deploy/models/mobilenet_v4_ema_demo_finetune_balanced_driver_distraction.om}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
INPUT_SHAPE="${INPUT_SHAPE:-input:1,3,224,224}"
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}"

if [[ "${ONNX_PATH}" != /* ]]; then
  ONNX_PATH="${ROOT_DIR}/${ONNX_PATH}"
fi
if [[ "${OM_PATH}" != /* ]]; then
  OM_PATH="${ROOT_DIR}/${OM_PATH}"
fi

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
  echo "ERROR: atc not found. Source the Ascend Toolkit environment first." >&2
  exit 1
fi

if [ ! -f "${ONNX_PATH}" ]; then
  echo "ERROR: ONNX model not found: ${ONNX_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OM_PATH}")"

echo "Converting MobileNetV4 EMA balanced model:"
echo "  ONNX: ${ONNX_PATH}"
echo "  OM:   ${OM_PATH}"
echo "  SoC:  ${SOC_VERSION}"
echo "  Input: ${INPUT_SHAPE}"

atc \
  --model="${ONNX_PATH}" \
  --framework=5 \
  --output="${OM_PATH%.om}" \
  --input_format=NCHW \
  --input_shape="${INPUT_SHAPE}" \
  --soc_version="${SOC_VERSION}" \
  --precision_mode="${PRECISION_MODE}" \
  --log=info

if [ ! -f "${OM_PATH}" ]; then
  echo "ERROR: ATC finished but the OM file was not created: ${OM_PATH}" >&2
  exit 1
fi

echo "OM model generated successfully: ${OM_PATH}"
