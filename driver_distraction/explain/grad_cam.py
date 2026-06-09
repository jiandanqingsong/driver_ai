"""Minimal Grad-CAM implementation for CNN classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from driver_distraction.data.transforms import build_eval_transform


@dataclass(frozen=True)
class GradCAMResult:
    overlay_path: Path
    heatmap_path: Path
    predicted_class: int
    target_class: int
    confidence: float


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = int(probs.argmax(dim=1).item())
        if target_class is None:
            target_class = predicted_class
        confidence = float(probs[:, predicted_class].item())
        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, predicted_class, confidence


def find_default_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise ValueError("Cannot infer target layer. Pass a CNN layer explicitly.")


def generate_gradcam_overlay(
    model: nn.Module,
    image_path: str | Path,
    output_path: str | Path,
    input_size: int,
    device: torch.device,
    target_class: int | None = None,
    alpha: float = 0.45,
    heatmap_output_path: str | Path | None = None,
) -> GradCAMResult:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path = Path(heatmap_output_path) if heatmap_output_path is not None else output_path.with_name(
        f"{output_path.stem}_heatmap{output_path.suffix}"
    )
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = Image.open(image_path).convert("RGB")
    transform = build_eval_transform(input_size)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    cam_runner = GradCAM(model, find_default_target_layer(model))
    try:
        cam, predicted_class, confidence = cam_runner(image_tensor, target_class)
    finally:
        cam_runner.close()

    raw = np.array(pil_image)
    raw_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    cam_resized = cv2.resize(cam, (raw_bgr.shape[1], raw_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_bgr, 1.0 - alpha, heatmap, alpha, 0)
    cv2.imwrite(str(output_path), overlay)
    cv2.imwrite(str(heatmap_path), heatmap)

    return GradCAMResult(
        overlay_path=output_path,
        heatmap_path=heatmap_path,
        predicted_class=predicted_class,
        target_class=predicted_class if target_class is None else target_class,
        confidence=confidence,
    )
