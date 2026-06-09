from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import StateFarmDataset, build_loader_kwargs
from driver_distraction.data.transforms import build_eval_transform
from driver_distraction.engine.evaluator import evaluate_model
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a classifier on an arbitrary manifest file.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--manifest", required=True, help="Manifest format: image_path label subject")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="mobilenet_v3_large", choices=SUPPORTED_MODELS)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--include-all-classes",
        action="store_true",
        help="Report macro metrics over all configured classes instead of labels present in this manifest.",
    )
    return parser.parse_args()


def resolve_device(device_name: str | None, config: dict) -> torch.device:
    requested = device_name or str(config["project"].get("device", "cuda"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def select_label_indices(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, include_all: bool) -> list[int]:
    if include_all:
        return list(range(num_classes))

    labels = sorted(set(map(int, y_true.tolist())) | set(map(int, y_pred.tolist())))
    return labels if labels else list(range(num_classes))


def display_label_names(class_names: list[str], label_indices: list[int]) -> list[str]:
    return [f"c{label}_{class_names[label]}" for label in label_indices]


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_indices: list[int]) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_indices,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_indices: list[int],
    label_names: list[str],
    output_dir: Path,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=label_indices)
    display_matrix = matrix.astype(float)
    row_sum = display_matrix.sum(axis=1, keepdims=True)
    display_matrix = np.divide(display_matrix, np.maximum(row_sum, 1.0))

    plt.figure(figsize=(max(8, len(label_names) * 1.3), max(6, len(label_names) * 1.1)))
    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    rows = ["," + ",".join(label_names)]
    for label_name, row in zip(label_names, matrix):
        rows.append(f"{label_name}," + ",".join(str(int(value)) for value in row))
    (output_dir / "confusion_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return matrix


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    class_names = list(config["data"]["class_names"])
    input_size = int(config["data"]["input_size"])
    device = resolve_device(args.device, config)

    dataset = StateFarmDataset(
        manifest_path=args.manifest,
        split=args.split,
        image_size=input_size,
        transform=build_eval_transform(input_size),
    )
    loader_cfg = config["data"]
    dataloader = DataLoader(
        dataset,
        **build_loader_kwargs(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            drop_last=False,
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
            prefetch_factor=loader_cfg.get("prefetch_factor", 1),
        ),
    )

    model = build_model(args.model, int(config["data"]["num_classes"]), pretrained=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))

    result = evaluate_model(model, dataloader, device, nn.CrossEntropyLoss())
    label_indices = select_label_indices(
        result.labels,
        result.predictions,
        int(config["data"]["num_classes"]),
        include_all=args.include_all_classes,
    )
    label_names = display_label_names(class_names, label_indices)
    output_dir = Path(
        args.output_dir
        or Path(config["project"]["output_dir"]) / "reports" / args.model / Path(args.manifest).stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        result.labels,
        result.predictions,
        labels=label_indices,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_outputs(result.labels, result.predictions, label_indices, label_names, output_dir)

    summary = summarize_metrics(result.labels, result.predictions, label_indices)
    summary["loss"] = float(result.loss) if result.loss is not None else None
    summary["model"] = args.model
    summary["manifest"] = str(args.manifest)
    summary["checkpoint"] = str(args.checkpoint)
    summary["num_samples"] = int(len(dataset))
    summary["label_indices"] = label_indices
    summary["label_names"] = label_names
    summary["metric_scope"] = "all_classes" if args.include_all_classes else "present_true_or_predicted_labels"
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report)
    print(
        f"{args.model} manifest={args.manifest} "
        f"scope={summary['metric_scope']} "
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
