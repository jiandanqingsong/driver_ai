"""Model builders for driver distraction classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ("mobilenet_v3_large", "resnet18", "mobilenet_v4")
MODEL_ALIASES = {
    "mobilenetv3": "mobilenet_v3_large",
    "mobilenet_v3": "mobilenet_v3_large",
    "mobilenet-v3-large": "mobilenet_v3_large",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "resnet-18": "resnet18",
    "resnet_18": "resnet18",
    "resnet18": "resnet18",
    "mobilenetv4": "mobilenet_v4",
    "mobilenet_v4": "mobilenet_v4",
    "mobilenet-v4": "mobilenet_v4",
    "mobilenetv4_conv_medium": "mobilenet_v4",
}

# Pure-conv MobileNetV4 variant from timm. Attention-free, so it exports cleanly
# to ONNX and converts to an Ascend OM model. Change here to swap variants
# (e.g. "mobilenetv4_conv_small" or "mobilenetv4_conv_large").
MOBILENET_V4_TIMM_NAME = "mobilenetv4_conv_medium"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    num_classes: int
    total_parameters: int
    trainable_parameters: int


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {model_name}. Choose one of {SUPPORTED_MODELS}.")
    return MODEL_ALIASES[key]


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _linear_head(in_features: int, num_classes: int, dropout: float = 0.0) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build ResNet18 and replace the final classifier."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        _set_trainable(model, False)

    in_features = model.fc.in_features
    model.fc = _linear_head(in_features, num_classes, dropout)
    _set_trainable(model.fc, True)
    return model


def build_mobilenet_v3_large(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
) -> nn.Module:
    """Build MobileNetV3-Large and replace the final classifier."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        _set_trainable(model.features, False)

    if dropout is not None:
        for layer in model.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    _set_trainable(model.classifier, True)
    return model


def build_mobilenet_v4(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
    drop_path_rate: float | None = None,
) -> nn.Module:
    """Build MobileNetV4 (pure-conv variant) from timm and replace the head.

    ``drop_path_rate`` enables stochastic depth, a strong regularizer for these
    timm backbones that helps close the train/val gap on the driver-split data.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "MobileNetV4 requires the 'timm' package. Install it with: pip install timm"
        ) from exc

    model = timm.create_model(
        MOBILENET_V4_TIMM_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=float(dropout or 0.0),
        drop_path_rate=float(drop_path_rate or 0.0),
    )

    if freeze_backbone:
        _set_trainable(model, False)
        _set_trainable(model.get_classifier(), True)

    return model


def build_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
    drop_path_rate: float | None = None,
) -> nn.Module:
    """Build a supported classifier by name.

    ``drop_path_rate`` only applies to models that support stochastic depth
    (currently ``mobilenet_v4``); it is ignored by the others.
    """
    model_name = normalize_model_name(model_name)

    if model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=float(dropout or 0.0),
        )

    if model_name == "mobilenet_v3_large":
        return build_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )

    if model_name == "mobilenet_v4":
        return build_mobilenet_v4(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

    raise AssertionError(f"Unhandled normalized model name: {model_name}")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def describe_model(model: nn.Module, name: str, num_classes: int) -> ModelInfo:
    return ModelInfo(
        name=normalize_model_name(name),
        num_classes=num_classes,
        total_parameters=count_parameters(model, trainable_only=False),
        trainable_parameters=count_parameters(model, trainable_only=True),
    )


@torch.inference_mode()
def check_forward_shape(model: nn.Module, input_size: int = 224, batch_size: int = 2) -> tuple[int, ...]:
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    logits = model(dummy)
    return tuple(logits.shape)
