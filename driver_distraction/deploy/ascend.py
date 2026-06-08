"""Ascend ATC command generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_atc_command(config: dict[str, Any]) -> list[str]:
    export_cfg = config["export"]
    ascend_cfg = config["ascend"]
    onnx_path = Path(export_cfg["onnx_path"])
    om_path = Path(ascend_cfg["om_path"])

    return [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={om_path.with_suffix('')}",
        f"--input_shape={ascend_cfg['input_shape']}",
        f"--soc_version={ascend_cfg['soc_version']}",
        f"--precision_mode={ascend_cfg['precision_mode']}",
        "--log=info",
    ]
