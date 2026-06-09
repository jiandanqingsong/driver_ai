"""Training loop for driver distraction classification."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from driver_distraction.engine.evaluator import evaluate_model
from driver_distraction.utils.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class FitResult:
    best_acc: float
    best_epoch: int
    last_epoch: int
    checkpoint_dir: str
    best_metric: str = "val_acc"
    best_metric_value: float = 0.0


def batch_to_device(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract images and labels from a batch and move them to device.

    Datasets in this project return `(image, label, image_path, subject)`, while
    some quick tests may only return `(image, label)`.
    """
    images, labels = batch[:2]
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    train_cfg = config["train"]
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg["epochs"])

    if scheduler_name in {"none", ""}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_criterion(config: dict[str, Any], device: torch.device | None = None) -> nn.Module:
    train_cfg = config["train"]
    class_weights = train_cfg.get("class_weights")
    weight_tensor = None

    if class_weights:
        class_names = list(config["data"]["class_names"])
        if isinstance(class_weights, dict):
            weights = [float(class_weights.get(class_name, 1.0)) for class_name in class_names]
        else:
            weights = [float(value) for value in class_weights]
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )


def build_grad_scaler(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def compute_macro_f1(labels, predictions, num_classes: int) -> float:
    labels_to_score = sorted(set(map(int, labels.tolist())) | set(map(int, predictions.tolist())))
    if not labels_to_score:
        return 0.0
    return float(
        f1_score(
            labels,
            predictions,
            labels=labels_to_score,
            average="macro",
            zero_division=0,
        )
    )


def select_best_metric(metric_name: str, val_acc: float, val_macro_f1: float) -> float:
    metric_name = metric_name.lower()
    if metric_name in {"val_acc", "accuracy", "acc"}:
        return val_acc
    if metric_name in {"val_macro_f1", "macro_f1", "f1"}:
        return val_macro_f1
    raise ValueError(f"Unsupported save_best_metric: {metric_name}")


def append_history_row(history_path: str | Path, row: dict[str, Any]) -> None:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_training_state(
    resume_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> tuple[int, float, int, float]:
    checkpoint = load_checkpoint(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_metric_value = float(checkpoint.get("best_metric_value", best_acc))
    return start_epoch, best_acc, best_epoch, best_metric_value


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    scaler=None,
    epoch: int | None = None,
) -> dict[str, float]:
    model.train()
    scaler = scaler or build_grad_scaler(device, use_amp)
    total_loss = 0.0
    correct = 0
    total_seen = 0
    desc = "train" if epoch is None else f"train epoch {epoch}"

    for batch in tqdm(dataloader, desc=desc, leave=False):
        images, labels = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.detach().argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((preds == labels).sum().item())
        total_seen += labels.size(0)

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": correct / max(total_seen, 1),
    }


def fit(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    resume_path: str | Path | None = None,
) -> FitResult:
    train_cfg = config["train"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(train_cfg.get("history_path", checkpoint_dir / "history.csv"))

    criterion = build_criterion(config, device=device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    use_amp = bool(train_cfg.get("amp", True))
    scaler = build_grad_scaler(device, use_amp)
    epochs = int(train_cfg["epochs"])
    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0
    best_metric = str(train_cfg.get("save_best_metric", "val_acc"))
    best_metric_value = -1.0

    if resume_path is not None:
        start_epoch, best_acc, best_epoch, best_metric_value = load_training_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(
            f"Resumed from {resume_path}, start_epoch={start_epoch}, "
            f"best_acc={best_acc:.4f}, best_metric_value={best_metric_value:.4f}"
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            epoch=epoch,
        )
        val_result = evaluate_model(model, dataloaders["val"], device, criterion)
        val_macro_f1 = compute_macro_f1(
            val_result.labels,
            val_result.predictions,
            int(config["data"]["num_classes"]),
        )
        if scheduler is not None:
            scheduler.step()

        current_metric_value = select_best_metric(best_metric, val_result.accuracy, val_macro_f1)
        is_best = current_metric_value > best_metric_value
        if is_best:
            best_metric_value = current_metric_value
            best_acc = val_result.accuracy
            best_epoch = epoch

        epoch_seconds = time.time() - epoch_start
        state = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value,
            "config": config,
        }
        save_checkpoint(state, checkpoint_dir / "last.pt")
        if is_best:
            save_checkpoint(state, checkpoint_dir / "best.pt")

        history_row = {
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_result.loss,
            "val_acc": val_result.accuracy,
            "val_macro_f1": val_macro_f1,
            "best_acc": best_acc,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
            "epoch_seconds": epoch_seconds,
        }
        append_history_row(history_path, history_row)

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"lr={history_row['lr']:.6g} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"best_{best_metric}={best_metric_value:.4f} time={epoch_seconds:.1f}s"
        )

    return FitResult(
        best_acc=best_acc,
        best_epoch=best_epoch,
        last_epoch=epochs,
        checkpoint_dir=str(checkpoint_dir),
        best_metric=best_metric,
        best_metric_value=best_metric_value,
    )
