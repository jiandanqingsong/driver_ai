"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvalResult:
    loss: float | None
    accuracy: float
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []

    for images, labels in tqdm(dataloader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)

        total_seen += labels.size(0)
        correct += int((preds == labels).sum().item())
        labels_all.append(labels.detach().cpu().numpy())
        preds_all.append(preds.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())

    loss_value = total_loss / total_seen if criterion is not None and total_seen else None
    return EvalResult(
        loss=loss_value,
        accuracy=correct / max(total_seen, 1),
        labels=np.concatenate(labels_all) if labels_all else np.array([]),
        predictions=np.concatenate(preds_all) if preds_all else np.array([]),
        probabilities=np.concatenate(probs_all) if probs_all else np.array([]),
    )
