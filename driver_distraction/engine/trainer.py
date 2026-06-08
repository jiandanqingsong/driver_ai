"""Training loop for driver distraction classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from driver_distraction.engine.evaluator import evaluate_model
from driver_distraction.utils.checkpoint import save_checkpoint


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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    total_loss = 0.0
    correct = 0
    total_seen = 0

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
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
) -> dict[str, float]:
    train_cfg = config["train"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    use_amp = bool(train_cfg.get("amp", True))
    epochs = int(train_cfg["epochs"])
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, use_amp)
        val_result = evaluate_model(model, dataloaders["val"], device, criterion)
        if scheduler is not None:
            scheduler.step()

        is_best = val_result.accuracy > best_acc
        if is_best:
            best_acc = val_result.accuracy
            best_epoch = epoch

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": config,
        }
        save_checkpoint(state, checkpoint_dir / "last.pt")
        if is_best:
            save_checkpoint(state, checkpoint_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}"
        )

    return {"best_acc": best_acc, "best_epoch": float(best_epoch)}
