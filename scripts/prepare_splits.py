from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.splits import apply_split, load_metadata, make_driver_split, save_split, save_split_manifests
from driver_distraction.data.statefarm import resolve_data_paths
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare State Farm driver-id split.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, image_root, metadata_csv = resolve_data_paths(config)
    split_cfg = config["data"]["split"]
    df = load_metadata(metadata_csv)
    split = make_driver_split(
        subjects=df["subject"].astype(str).tolist(),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["random_seed"]),
        explicit_val_drivers=split_cfg.get("explicit_val_drivers") or [],
        explicit_test_drivers=split_cfg.get("explicit_test_drivers") or [],
    )
    save_split(split, split_cfg["split_file"])
    frames = apply_split(df, split)
    manifest_dir = split_cfg.get("manifest_dir", Path(split_cfg["split_file"]).parent)
    manifest_paths = save_split_manifests(frames, manifest_dir, image_root)
    print(f"Saved split to {split_cfg['split_file']}")
    print(split.to_dict())
    for split_name, path in manifest_paths.items():
        drivers = frames[split_name]["subject"].nunique()
        images = len(frames[split_name])
        print(f"{split_name}: {images} images, {drivers} drivers -> {path}")


if __name__ == "__main__":
    main()
