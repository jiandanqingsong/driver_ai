"""State Farm dataset and DataLoader builders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal

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


SplitName = Literal["train", "val", "test"]


class StateFarmDataset(Dataset):
    """State Farm dataset that returns image tensor, label, image path and subject.

    The preferred manifest format is one whitespace-separated sample per line:

        image_path label subject
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        split: SplitName = "train",
        image_size: int = 224,
        transform: Callable | None = None,
        root_dir: str | Path | None = None,
        samples: list[tuple[str, int, str]] | None = None,
    ) -> None:
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform if transform is not None else _build_transform(split, image_size)

        if samples is not None:
            self.samples = samples
        elif manifest_path is not None:
            self.samples = self._load_manifest(Path(manifest_path))
        else:
            raise ValueError("Either manifest_path or samples must be provided.")

        if not self.samples:
            raise ValueError("StateFarmDataset received no samples.")

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        image_root: str | Path,
        split: SplitName,
        image_size: int = 224,
        transform: Callable | None = None,
    ) -> "StateFarmDataset":
        image_root = Path(image_root)
        samples: list[tuple[str, int, str]] = []

        for row in frame.reset_index(drop=True).itertuples(index=False):
            classname = str(row.classname)
            image_path = image_root / classname / str(row.img)
            label = STATE_FARM_CLASS_TO_IDX[classname]
            subject = str(row.subject)
            samples.append((str(image_path), label, subject))

        return cls(samples=samples, split=split, image_size=image_size, transform=transform)

    def _load_manifest(self, manifest_path: Path) -> list[tuple[str, int, str]]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        samples: list[tuple[str, int, str]] = []
        with manifest_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid manifest line {line_number} in {manifest_path}: "
                        "expected 'image_path label subject'."
                    )

                image_path, label_text, subject = parts
                try:
                    label = int(label_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label at line {line_number} in {manifest_path}: {label_text}"
                    ) from exc
                samples.append((image_path, label, subject))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, subject = self.samples[index]
        resolved_path = Path(image_path)
        if self.root_dir is not None and not resolved_path.is_absolute():
            resolved_path = self.root_dir / resolved_path

        image = Image.open(resolved_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, label, str(resolved_path), subject


def _build_transform(split: SplitName, image_size: int, augmentation: dict[str, Any] | None = None):
    if split == "train":
        return build_train_transform(image_size, augmentation)
    if split in {"val", "test"}:
        return build_eval_transform(image_size)
    raise ValueError(f"Unsupported split: {split}. Expected one of: train, val, test.")


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
    augmentation = config["data"].get("augmentation", {})

    return {
        "train": StateFarmDataset.from_dataframe(
            frames["train"],
            image_root,
            "train",
            input_size,
            transform=build_train_transform(input_size, augmentation),
        ),
        "val": StateFarmDataset.from_dataframe(frames["val"], image_root, "val", input_size),
        "test": StateFarmDataset.from_dataframe(frames["test"], image_root, "test", input_size),
    }


def build_dataloader(
    manifest_path: str | Path,
    split: SplitName,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    drop_last: bool | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    root_dir: str | Path | None = None,
) -> DataLoader:
    dataset = StateFarmDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
        root_dir=root_dir,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    loader_kwargs = build_loader_kwargs(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return DataLoader(dataset, **loader_kwargs)


def build_loader_kwargs(
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
) -> dict[str, Any]:
    """Build DataLoader kwargs with Windows-safe multiprocessing behavior."""
    num_workers = int(num_workers)
    if os.name == "nt" and num_workers < 0:
        num_workers = 0

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

    return kwargs


def build_split_loader(dataset: StateFarmDataset, split: SplitName, batch_size: int, data_cfg: dict[str, Any]) -> DataLoader:
    shuffle = split == "train"
    drop_last = split == "train"
    loader_kwargs = build_loader_kwargs(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=drop_last,
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_dataloaders(config: dict[str, Any], batch_size: int | None = None) -> dict[str, DataLoader]:
    datasets = build_datasets(config)
    data_cfg = config["data"]
    train_cfg = config["train"]
    bs = int(batch_size or train_cfg["batch_size"])

    return {
        "train": build_split_loader(datasets["train"], "train", bs, data_cfg),
        "val": build_split_loader(datasets["val"], "val", bs, data_cfg),
        "test": build_split_loader(datasets["test"], "test", bs, data_cfg),
    }
