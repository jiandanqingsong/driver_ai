from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.data.statefarm import build_dataloaders
from driver_distraction.engine.trainer import fit
from driver_distraction.models.factory import SUPPORTED_MODELS, build_model, describe_model
from driver_distraction.utils.config import load_config, save_config
from driver_distraction.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train driver distraction classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume from a checkpoint path or last.pt.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_name)
        print("CUDA requested but not available. Falling back to CPU.")
    return torch.device("cpu")


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    train_cfg = config["train"]
    data_cfg = config["data"]

    if args.model is not None:
        train_cfg["model_name"] = args.model

    if args.device is not None:
        config["project"]["device"] = args.device
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        data_cfg["pin_memory"] = args.pin_memory
    if args.persistent_workers is not None:
        data_cfg["persistent_workers"] = args.persistent_workers
    if args.prefetch_factor is not None:
        data_cfg["prefetch_factor"] = args.prefetch_factor
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.pretrained is not None:
        train_cfg["pretrained"] = args.pretrained
    if args.amp is not None:
        train_cfg["amp"] = args.amp
    if args.freeze_backbone:
        train_cfg["freeze_backbone"] = True
    if args.dropout is not None:
        train_cfg["dropout"] = args.dropout

    base_checkpoint_dir = Path(args.checkpoint_dir or train_cfg["checkpoint_dir"])
    if args.checkpoint_dir is None:
        base_checkpoint_dir = base_checkpoint_dir / str(train_cfg["model_name"])
    train_cfg["checkpoint_dir"] = str(base_checkpoint_dir)
    if not train_cfg.get("history_path") or args.model is not None or args.checkpoint_dir is not None:
        train_cfg["history_path"] = str(base_checkpoint_dir / "history.csv")

    return config


def resolve_resume_path(resume_arg: str | None, checkpoint_dir: str | Path) -> Path | None:
    if resume_arg is None:
        return None

    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg == "auto":
        candidate = checkpoint_dir / "last.pt"
        if candidate.exists():
            return candidate
        print(f"No checkpoint found for auto resume: {candidate}")
        return None

    return Path(resume_arg)


def configure_backend(device: torch.device, deterministic: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    elif hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    deterministic = bool(config["train"].get("deterministic", True))
    seed_everything(int(config["project"]["seed"]), deterministic=deterministic)
    device = resolve_device(str(config["project"].get("device", "cuda")))
    configure_backend(device, deterministic)

    checkpoint_dir = Path(config["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / "config.yaml")

    dataloaders = build_dataloaders(config)
    model = build_model(
        model_name=config["train"]["model_name"],
        num_classes=int(config["data"]["num_classes"]),
        pretrained=bool(config["train"]["pretrained"]),
        freeze_backbone=bool(config["train"].get("freeze_backbone", False)),
        dropout=config["train"].get("dropout"),
    ).to(device)

    model_info = describe_model(model, config["train"]["model_name"], int(config["data"]["num_classes"]))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Model: {model_info.name}, classes={model_info.num_classes}, "
        f"total_params={model_info.total_parameters}, trainable_params={model_info.trainable_parameters}"
    )
    print(
        f"Dataset: train={len(dataloaders['train'].dataset)}, "
        f"val={len(dataloaders['val'].dataset)}, test={len(dataloaders['test'].dataset)}"
    )
    print(f"Checkpoints: {checkpoint_dir}")

    resume_arg = args.resume if args.resume is not None else config["train"].get("resume")
    resume_path = resolve_resume_path(resume_arg, checkpoint_dir)
    result = fit(model, dataloaders, config, device, resume_path=resume_path)
    print(f"Training finished: {result}")


if __name__ == "__main__":
    main()
