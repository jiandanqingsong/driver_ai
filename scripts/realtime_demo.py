from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.realtime.camera_demo import run_camera_demo
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime driver distraction demo.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", default=None, help="Camera index or video path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_camera_demo(config, source=args.source)


if __name__ == "__main__":
    main()
