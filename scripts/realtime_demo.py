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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-decision-filter", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--no-window", dest="show_window", action="store_false")
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.set_defaults(show_window=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    realtime_cfg = config["realtime"]
    if args.checkpoint is not None:
        realtime_cfg["checkpoint"] = args.checkpoint
    if args.model is not None:
        realtime_cfg["model_name"] = args.model
    if args.device is not None:
        config["project"]["device"] = args.device
    if args.confidence_threshold is not None:
        realtime_cfg["confidence_threshold"] = args.confidence_threshold
    if args.ema_alpha is not None:
        realtime_cfg.setdefault("temporal_smoothing", {})["alpha"] = args.ema_alpha
    if args.no_smoothing:
        realtime_cfg.setdefault("temporal_smoothing", {})["enabled"] = False
    if args.no_decision_filter:
        realtime_cfg.setdefault("decision_filter", {})["enabled"] = False
    if args.no_voice:
        realtime_cfg.setdefault("voice", {})["enabled"] = False
    if args.show_window is not None:
        realtime_cfg["show_window"] = args.show_window
    if args.save_video is not None:
        realtime_cfg["save_video_path"] = args.save_video
    if args.camera_width is not None:
        realtime_cfg.setdefault("camera", {})["width"] = args.camera_width
    if args.camera_height is not None:
        realtime_cfg.setdefault("camera", {})["height"] = args.camera_height
    if args.camera_fps is not None:
        realtime_cfg.setdefault("camera", {})["fps"] = args.camera_fps
    if args.max_frames is not None:
        realtime_cfg["max_frames"] = args.max_frames
    run_camera_demo(config, source=args.source)


if __name__ == "__main__":
    main()
