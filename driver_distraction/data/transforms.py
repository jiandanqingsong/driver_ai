"""Image transforms for training, evaluation and realtime inference."""

from __future__ import annotations

from typing import Any

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pair(value, default):
    if value is None:
        return default
    return tuple(value)


def build_train_transform(input_size: int, augmentation: dict[str, Any] | None = None):
    augmentation = augmentation or {}
    resize_size = int(augmentation.get("resize_size", 256))
    crop_scale = _pair(augmentation.get("random_resized_crop_scale"), (0.85, 1.0))
    crop_ratio = _pair(augmentation.get("random_resized_crop_ratio"), (0.9, 1.1))
    jitter_cfg = augmentation.get("color_jitter", {})
    rotation_degrees = float(augmentation.get("rotation_degrees", 5))

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(input_size, scale=crop_scale, ratio=crop_ratio),
            transforms.ColorJitter(
                brightness=float(jitter_cfg.get("brightness", 0.1)),
                contrast=float(jitter_cfg.get("contrast", 0.1)),
                saturation=float(jitter_cfg.get("saturation", 0.1)),
                hue=float(jitter_cfg.get("hue", 0.02)),
            ),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(input_size: int, resize_size: int = 256):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_realtime_transform(input_size: int):
    return build_eval_transform(input_size)
