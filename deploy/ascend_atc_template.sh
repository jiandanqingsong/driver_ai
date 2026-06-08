#!/usr/bin/env bash
set -euo pipefail

CONFIG_ONNX_PATH="outputs/export/driver_distraction.onnx"
CONFIG_OM_PATH="outputs/export/driver_distraction.om"
SOC_VERSION="Ascend310P3"
INPUT_SHAPE="input:1,3,224,224"
PRECISION_MODE="allow_fp32_to_fp16"

mkdir -p "$(dirname "${CONFIG_OM_PATH}")"

atc \
  --model="${CONFIG_ONNX_PATH}" \
  --framework=5 \
  --output="${CONFIG_OM_PATH%.om}" \
  --input_shape="${INPUT_SHAPE}" \
  --soc_version="${SOC_VERSION}" \
  --precision_mode="${PRECISION_MODE}" \
  --log=info
