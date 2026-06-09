"""Driver-id based split utilities for the State Farm dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from driver_distraction.constants import STATE_FARM_CLASS_TO_IDX


@dataclass(frozen=True)
class DriverSplit:
    train_drivers: list[str]
    val_drivers: list[str]
    test_drivers: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train": self.train_drivers,
            "val": self.val_drivers,
            "test": self.test_drivers,
        }


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"subject", "classname", "img"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata csv missing columns: {sorted(missing)}")
    return df


def make_driver_split(
    subjects: Iterable[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    explicit_val_drivers: Iterable[str] | None = None,
    explicit_test_drivers: Iterable[str] | None = None,
) -> DriverSplit:
    all_drivers = sorted(set(str(s) for s in subjects))
    val_drivers = sorted(set(explicit_val_drivers or []))
    test_drivers = sorted(set(explicit_test_drivers or []))

    overlap = set(val_drivers).intersection(test_drivers)
    if overlap:
        raise ValueError(f"Drivers cannot be both val and test: {sorted(overlap)}")

    unknown = set(val_drivers + test_drivers).difference(all_drivers)
    if unknown:
        raise ValueError(f"Explicit drivers not found in metadata: {sorted(unknown)}")

    remaining = [driver for driver in all_drivers if driver not in val_drivers and driver not in test_drivers]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    if not test_drivers:
        test_count = max(1, int(round(len(all_drivers) * test_ratio)))
        test_drivers = sorted(remaining[:test_count])
        remaining = remaining[test_count:]

    if not val_drivers:
        val_count = max(1, int(round(len(all_drivers) * val_ratio)))
        val_drivers = sorted(remaining[:val_count])
        remaining = remaining[val_count:]

    train_drivers = sorted(driver for driver in remaining if driver not in val_drivers and driver not in test_drivers)
    if not train_drivers:
        raise ValueError("Driver split produced an empty training set.")

    return DriverSplit(train_drivers=train_drivers, val_drivers=val_drivers, test_drivers=test_drivers)


def apply_split(df: pd.DataFrame, split: DriverSplit) -> dict[str, pd.DataFrame]:
    split_map = split.to_dict()
    return {
        name: df[df["subject"].astype(str).isin(drivers)].reset_index(drop=True)
        for name, drivers in split_map.items()
    }


def save_split(split: DriverSplit, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")


def load_split(path: str | Path) -> DriverSplit:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DriverSplit(
        train_drivers=list(data["train"]),
        val_drivers=list(data["val"]),
        test_drivers=list(data["test"]),
    )


def save_split_manifests(
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    image_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_dir)
    saved_paths: dict[str, Path] = {}

    for split_name, frame in frames.items():
        manifest = frame.copy()
        manifest["label"] = manifest["classname"].map(STATE_FARM_CLASS_TO_IDX)
        manifest["image_path"] = manifest.apply(
            lambda row: str(image_dir / str(row["classname"]) / str(row["img"])),
            axis=1,
        )

        output_path = output_dir / f"{split_name}.txt"
        with output_path.open("w", encoding="utf-8") as file:
            for row in manifest.itertuples(index=False):
                file.write(f"{row.image_path} {int(row.label)} {row.subject}\n")

        saved_paths[split_name] = output_path

    return saved_paths
