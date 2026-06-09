from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import StateFarmDataset, build_loader_kwargs
from driver_distraction.data.transforms import build_eval_transform, build_train_transform
from driver_distraction.engine.trainer import fit
from driver_distraction.models.factory import build_model, describe_model
from driver_distraction.utils.checkpoint import load_checkpoint
from driver_distraction.utils.config import load_config, save_config
from driver_distraction.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3-Large on self-collected demo-scene data.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--demo-root", default="data/demo_scene")
    parser.add_argument("--base-checkpoint", default="outputs/checkpoints/mobilenet_v3_large/best.pt")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/mobilenet_v3_large_demo_finetune")
    parser.add_argument("--manifest-dir", default="outputs/demo_scene")
    parser.add_argument("--model", default="mobilenet_v3_large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--demo-repeat", type=int, default=3, help="Repeat demo train samples in mixed training manifest.")
    parser.add_argument("--replay-manifest", default="outputs/splits/train.txt")
    parser.add_argument("--replay-per-class", type=int, default=80)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def scan_demo_samples(demo_root: str | Path) -> dict[int, list[str]]:
    demo_root = Path(demo_root)
    samples_by_label: dict[int, list[str]] = defaultdict(list)
    for class_dir in sorted(demo_root.glob("c*")):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name[1:])
        except ValueError:
            continue
        for path in sorted(class_dir.glob("*.jpg")):
            samples_by_label[label].append(f"{path.as_posix()} {label} demo_{class_dir.name}")
    return samples_by_label


def split_demo_samples(
    samples_by_label: dict[int, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    train_lines: list[str] = []
    val_lines: list[str] = []

    for label, lines in sorted(samples_by_label.items()):
        lines = list(lines)
        rng.shuffle(lines)
        val_count = max(1, int(round(len(lines) * val_ratio)))
        val_lines.extend(lines[:val_count])
        train_lines.extend(lines[val_count:])

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines


def sample_replay_lines(
    replay_manifest: str | Path,
    per_class: int,
    seed: int,
) -> list[str]:
    replay_manifest = Path(replay_manifest)
    if per_class <= 0 or not replay_manifest.exists():
        return []

    by_label: dict[int, list[str]] = defaultdict(list)
    for line in replay_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        by_label[int(parts[1])].append(line)

    rng = random.Random(seed)
    replay_lines: list[str] = []
    for label, lines in sorted(by_label.items()):
        lines = list(lines)
        rng.shuffle(lines)
        replay_lines.extend(lines[: min(per_class, len(lines))])

    rng.shuffle(replay_lines)
    return replay_lines


def write_manifest(lines: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def build_demo_manifests(args: argparse.Namespace) -> dict[str, Path]:
    samples_by_label = scan_demo_samples(args.demo_root)
    if not samples_by_label:
        raise RuntimeError(f"No demo images found under {args.demo_root}")

    train_lines, val_lines = split_demo_samples(samples_by_label, args.val_ratio, args.seed)
    replay_lines = [] if args.no_replay else sample_replay_lines(args.replay_manifest, args.replay_per_class, args.seed)
    mixed_lines = train_lines * max(1, int(args.demo_repeat)) + replay_lines
    random.Random(args.seed).shuffle(mixed_lines)

    manifest_dir = Path(args.manifest_dir)
    paths = {
        "demo_train": write_manifest(train_lines, manifest_dir / "demo_train.txt"),
        "demo_val": write_manifest(val_lines, manifest_dir / "demo_val.txt"),
        "replay_train": write_manifest(replay_lines, manifest_dir / "replay_train.txt"),
        "mixed_train": write_manifest(mixed_lines, manifest_dir / "mixed_train.txt"),
    }

    print("Demo samples:")
    for label, lines in sorted(samples_by_label.items()):
        print(f"  c{label}: {len(lines)}")
    print(f"Train demo: {len(train_lines)}, val demo: {len(val_lines)}, replay: {len(replay_lines)}, mixed: {len(mixed_lines)}")
    return paths


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def resolve_resume_path(resume_arg: str | None, checkpoint_dir: str | Path) -> Path | None:
    if resume_arg is None:
        return None
    if resume_arg == "auto":
        candidate = Path(checkpoint_dir) / "last.pt"
        return candidate if candidate.exists() else None
    return Path(resume_arg)


def build_loaders(config: dict, paths: dict[str, Path], args: argparse.Namespace) -> dict[str, DataLoader]:
    input_size = int(config["data"]["input_size"])
    augmentation = config["data"].get("augmentation", {})
    train_dataset = StateFarmDataset(
        manifest_path=paths["mixed_train"],
        split="train",
        image_size=input_size,
        transform=build_train_transform(input_size, augmentation),
    )
    val_dataset = StateFarmDataset(
        manifest_path=paths["demo_val"],
        split="val",
        image_size=input_size,
        transform=build_eval_transform(input_size),
    )
    loader_cfg = config["data"]
    return {
        "train": DataLoader(
            train_dataset,
            **build_loader_kwargs(
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=bool(loader_cfg.get("pin_memory", True)),
                drop_last=False,
                persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
                prefetch_factor=loader_cfg.get("prefetch_factor", 1),
            ),
        ),
        "val": DataLoader(
            val_dataset,
            **build_loader_kwargs(
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=bool(loader_cfg.get("pin_memory", True)),
                drop_last=False,
                persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
                prefetch_factor=loader_cfg.get("prefetch_factor", 1),
            ),
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["project"]["device"] = args.device
    config["train"]["model_name"] = args.model
    config["train"]["checkpoint_dir"] = args.checkpoint_dir
    config["train"]["history_path"] = str(Path(args.checkpoint_dir) / "history.csv")
    config["train"]["epochs"] = args.epochs
    config["train"]["lr"] = args.lr
    config["train"]["weight_decay"] = args.weight_decay
    config["train"]["freeze_backbone"] = args.freeze_backbone
    config["train"]["save_best_metric"] = "val_macro_f1"
    config["train"]["class_weights"] = None

    seed_everything(args.seed, deterministic=bool(config["train"].get("deterministic", True)))
    paths = build_demo_manifests(args)
    device = resolve_device(args.device)
    dataloaders = build_loaders(config, paths, args)

    model = build_model(
        model_name=args.model,
        num_classes=int(config["data"]["num_classes"]),
        pretrained=False,
        freeze_backbone=args.freeze_backbone,
        dropout=config["train"].get("dropout"),
    ).to(device)

    base_checkpoint = load_checkpoint(args.base_checkpoint, map_location=device)
    model.load_state_dict(base_checkpoint.get("model_state", base_checkpoint))
    info = describe_model(model, args.model, int(config["data"]["num_classes"]))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Loaded base checkpoint: {args.base_checkpoint}")
    print(f"Model: {info.name}, total_params={info.total_parameters}, trainable_params={info.trainable_parameters}")
    print(f"Fine-tune checkpoints: {args.checkpoint_dir}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / "config.yaml")
    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    result = fit(model, dataloaders, config, device, resume_path=resume_path)
    print(f"Fine-tuning finished: {result}")


if __name__ == "__main__":
    main()
