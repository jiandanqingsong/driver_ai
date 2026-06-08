from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.explain.grad_cam import generate_gradcam_overlay
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlay for one image.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_path = args.image or config["gradcam"]["image_path"]
    checkpoint_path = args.checkpoint or config["gradcam"]["checkpoint"]
    output_path = args.output or config["gradcam"]["output_path"]
    target_class = args.target_class if args.target_class is not None else config["gradcam"]["target_class"]
    model_name = args.model or config["train"]["model_name"]

    device_name = str(config["project"].get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    model = build_model(model_name, int(config["data"]["num_classes"]), pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))

    saved_path, used_class = generate_gradcam_overlay(
        model=model,
        image_path=image_path,
        output_path=output_path,
        input_size=int(config["data"]["input_size"]),
        device=device,
        target_class=target_class,
        alpha=float(config["gradcam"]["alpha"]),
    )
    print(f"Grad-CAM saved to {saved_path}, target_class={used_class}")


if __name__ == "__main__":
    main()
