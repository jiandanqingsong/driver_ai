from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.realtime.web_demo import run_web_demo
from driver_distraction.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web dashboard for realtime driver distraction demo.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", default=None, help="Camera index, video path or image path.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-decision-filter", action="store_true")
    parser.add_argument("--no-voice", action="store_true", help="Disable server-side pyttsx3 voice alarm.")
    parser.add_argument("--browser-voice-default", action="store_true")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument(
        "--camera-backend",
        default=None,
        choices=("auto", "dshow", "msmf", "v4l2", "default"),
    )
    parser.add_argument("--camera-fourcc", default=None)
    parser.add_argument("--camera-buffer-size", type=int, default=None)
    parser.add_argument("--camera-startup-timeout", type=float, default=None)
    parser.add_argument("--camera-read-timeout", type=float, default=None)
    parser.add_argument("--threaded-camera", dest="threaded_camera", action="store_true", default=None)
    parser.add_argument("--no-threaded-camera", dest="threaded_camera", action="store_false")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Process one frame and exit for quick validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    realtime_cfg = config["realtime"]
    web_cfg = realtime_cfg.setdefault("web", {})

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
    if args.browser_voice_default:
        web_cfg["browser_voice_default"] = True
    if args.camera_width is not None:
        realtime_cfg.setdefault("camera", {})["width"] = args.camera_width
    if args.camera_height is not None:
        realtime_cfg.setdefault("camera", {})["height"] = args.camera_height
    if args.camera_fps is not None:
        realtime_cfg.setdefault("camera", {})["fps"] = args.camera_fps
    if args.camera_backend is not None:
        realtime_cfg.setdefault("camera", {})["backend"] = args.camera_backend
    if args.camera_fourcc is not None:
        realtime_cfg.setdefault("camera", {})["fourcc"] = args.camera_fourcc
    if args.camera_buffer_size is not None:
        realtime_cfg.setdefault("camera", {})["buffer_size"] = args.camera_buffer_size
    if args.camera_startup_timeout is not None:
        realtime_cfg.setdefault("camera", {})["startup_timeout"] = args.camera_startup_timeout
    if args.camera_read_timeout is not None:
        realtime_cfg.setdefault("camera", {})["read_timeout"] = args.camera_read_timeout
    if args.threaded_camera is not None:
        realtime_cfg.setdefault("camera", {})["threaded"] = args.threaded_camera
    if args.jpeg_quality is not None:
        web_cfg["jpeg_quality"] = args.jpeg_quality
    if args.max_frames is not None:
        realtime_cfg["max_frames"] = args.max_frames

    run_web_demo(
        config,
        source=args.source,
        host=args.host,
        port=args.port,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_path = ROOT / "outputs" / "web_demo_error.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
