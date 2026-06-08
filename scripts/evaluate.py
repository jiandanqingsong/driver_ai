from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import build_dataloaders
from driver_distraction.engine.evaluator import evaluate_model
from driver_distraction.engine.metrics import save_confusion_matrix, write_classification_report
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate driver distraction classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default=None, choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_name = args.split or config["eval"]["split"]
    checkpoint_path = args.checkpoint or config["eval"]["checkpoint"]
    model_name = args.model or config["train"]["model_name"]

    device_name = str(config["project"].get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    dataloaders = build_dataloaders(config, batch_size=int(config["eval"]["batch_size"]))

    model = build_model(model_name, int(config["data"]["num_classes"]), pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))

    result = evaluate_model(model, dataloaders[split_name], device, nn.CrossEntropyLoss())
    class_names = list(config["data"]["class_names"])
    report = write_classification_report(
        result.labels,
        result.predictions,
        class_names,
        config["eval"]["classification_report_path"],
    )
    save_confusion_matrix(
        result.labels,
        result.predictions,
        class_names,
        config["eval"]["confusion_matrix_path"],
        normalize=True,
    )
    print(report)
    print(f"{split_name} loss={result.loss:.4f} acc={result.accuracy:.4f}")


if __name__ == "__main__":
    main()
