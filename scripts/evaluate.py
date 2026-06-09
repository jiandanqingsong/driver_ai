from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import build_dataloaders
from driver_distraction.engine.evaluator import evaluate_model
from driver_distraction.engine.metrics import (
    save_confusion_matrix,
    save_confusion_matrix_csv,
    summarize_classification_metrics,
    write_classification_report,
)
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate driver distraction classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default=None, choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_name = args.split or config["eval"]["split"]
    checkpoint_path = args.checkpoint or config["eval"]["checkpoint"]
    model_name = args.model or config["train"]["model_name"]
    if args.batch_size is not None:
        config["eval"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    device_name = str(config["project"].get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    dataloaders = build_dataloaders(config, batch_size=int(config["eval"]["batch_size"]))

    model = build_model(model_name, int(config["data"]["num_classes"]), pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))

    result = evaluate_model(model, dataloaders[split_name], device, nn.CrossEntropyLoss())
    class_names = list(config["data"]["class_names"])
    output_dir = Path(args.output_dir or Path(config["project"]["output_dir"]) / "reports" / model_name / split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = write_classification_report(
        result.labels,
        result.predictions,
        class_names,
        output_dir / "classification_report.txt",
    )
    cm = save_confusion_matrix(
        result.labels,
        result.predictions,
        class_names,
        output_dir / "confusion_matrix.png",
        normalize=True,
    )
    save_confusion_matrix_csv(cm, class_names, output_dir / "confusion_matrix.csv")

    summary = summarize_classification_metrics(result.labels, result.predictions, class_names)
    summary["loss"] = float(result.loss) if result.loss is not None else None
    summary["model"] = model_name
    summary["split"] = split_name
    summary["checkpoint"] = str(checkpoint_path)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report)
    print(
        f"{model_name} {split_name} "
        f"loss={summary['loss']:.4f} "
        f"acc={summary['accuracy']:.4f} "
        f"macro_precision={summary['macro_precision']:.4f} "
        f"macro_recall={summary['macro_recall']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f} "
        f"weighted_f1={summary['weighted_f1']:.4f}"
    )
    print(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    main()
