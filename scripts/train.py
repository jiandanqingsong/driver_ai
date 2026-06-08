from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import build_dataloaders
from driver_distraction.engine.trainer import fit
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.config import load_config
from driver_distraction.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train driver distraction classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.model is not None:
        config["train"]["model_name"] = args.model
        config["train"]["checkpoint_dir"] = str(Path(config["train"]["checkpoint_dir"]) / args.model)

    seed_everything(int(config["project"]["seed"]))
    device_name = str(config["project"].get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    dataloaders = build_dataloaders(config)
    model = build_model(
        model_name=config["train"]["model_name"],
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(config["train"]["pretrained"]),
    ).to(device)

    result = fit(model, dataloaders, config, device)
    print(f"Training finished: {result}")


if __name__ == "__main__":
    main()
