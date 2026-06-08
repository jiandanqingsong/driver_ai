"""State Farm dataset and DataLoader builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from driver_distraction.constants import STATE_FARM_CLASS_TO_IDX
from driver_distraction.data.splits import (
    DriverSplit,
    apply_split,
    load_metadata,
    load_split,
    make_driver_split,
    save_split,
)
from driver_distraction.data.transforms import build_eval_transform, build_train_transform


class StateFarmDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: str | Path, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        classname = str(row["classname"])
        image_path = self.image_root / classname / str(row["img"])
        image = Image.open(image_path).convert("RGB")
        label = STATE_FARM_CLASS_TO_IDX[classname]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def resolve_data_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    image_root = root / data_cfg["image_dir"]
    metadata_csv = root / data_cfg["metadata_csv"]
    return root, image_root, metadata_csv


def build_or_load_split(config: dict[str, Any], df: pd.DataFrame) -> DriverSplit:
    split_cfg = config["data"]["split"]
    split_path = Path(split_cfg["split_file"])
    if split_path.exists():
        return load_split(split_path)

    split = make_driver_split(
        subjects=df["subject"].astype(str).tolist(),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["random_seed"]),
        explicit_val_drivers=split_cfg.get("explicit_val_drivers") or [],
        explicit_test_drivers=split_cfg.get("explicit_test_drivers") or [],
    )
    save_split(split, split_path)
    return split


def build_datasets(config: dict[str, Any]) -> dict[str, StateFarmDataset]:
    _, image_root, metadata_csv = resolve_data_paths(config)
    df = load_metadata(metadata_csv)
    split = build_or_load_split(config, df)
    frames = apply_split(df, split)
    input_size = int(config["data"]["input_size"])

    return {
        "train": StateFarmDataset(frames["train"], image_root, build_train_transform(input_size)),
        "val": StateFarmDataset(frames["val"], image_root, build_eval_transform(input_size)),
        "test": StateFarmDataset(frames["test"], image_root, build_eval_transform(input_size)),
    }


def build_dataloaders(config: dict[str, Any], batch_size: int | None = None) -> dict[str, DataLoader]:
    datasets = build_datasets(config)
    data_cfg = config["data"]
    train_cfg = config["train"]
    bs = int(batch_size or train_cfg["batch_size"])

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=bs,
            shuffle=True,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=bs,
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=bs,
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
        ),
    }
