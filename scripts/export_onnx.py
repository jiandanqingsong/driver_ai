from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export classifier checkpoint to ONNX.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["export"]["checkpoint"]
    output_path = Path(args.output or config["export"]["onnx_path"])
    model_name = args.model or config["train"]["model_name"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = build_model(model_name, int(config["data"]["num_classes"]), pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    input_size = int(config["data"]["input_size"])
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    input_name = str(config["export"]["input_name"])
    output_name = str(config["export"]["output_name"])
    dynamic_axes = None
    if bool(config["export"]["dynamic_batch"]):
        dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}

    export_kwargs = {
        "input_names": [input_name],
        "output_names": [output_name],
        "opset_version": int(config["export"]["opset_version"]),
        "dynamic_axes": dynamic_axes,
    }
    try:
        torch.onnx.export(model, dummy, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, output_path, **export_kwargs)
    print(f"ONNX exported to {output_path}")


if __name__ == "__main__":
    main()
