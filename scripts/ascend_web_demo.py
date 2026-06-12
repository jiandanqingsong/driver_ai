from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.realtime.ascend_web_demo import run_ascend_web_demo
from driver_distraction.utils.config import load_config


DEFAULT_MODEL = "deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the realtime web dashboard with an Ascend OM model."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the OM model.")
    parser.add_argument("--source", default=None, help="Camera index or video/image path.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-decision-filter", action="store_true")
    parser.add_argument("--browser-voice-default", action="store_true")
    parser.add_argument(
        "--server-voice",
        action="store_true",
        help="Enable pyttsx3 audio on the board. Browser voice remains available independently.",
    )
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "v4l2", "default"),
        default=None,
    )
    parser.add_argument("--camera-fourcc", default=None)
    parser.add_argument("--camera-buffer-size", type=int, default=None)
    parser.add_argument("--camera-startup-timeout", type=float, default=None)
    parser.add_argument("--camera-read-timeout", type=float, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    realtime_cfg = config["realtime"]
    web_cfg = realtime_cfg.setdefault("web", {})
    camera_cfg = realtime_cfg.setdefault("camera", {})

    if args.confidence_threshold is not None:
        realtime_cfg["confidence_threshold"] = args.confidence_threshold
    if args.ema_alpha is not None:
        realtime_cfg.setdefault("temporal_smoothing", {})["alpha"] = args.ema_alpha
    if args.no_smoothing:
        realtime_cfg.setdefault("temporal_smoothing", {})["enabled"] = False
    if args.no_decision_filter:
        realtime_cfg.setdefault("decision_filter", {})["enabled"] = False
    if args.camera_width is not None:
        camera_cfg["width"] = args.camera_width
    if args.camera_height is not None:
        camera_cfg["height"] = args.camera_height
    if args.camera_fps is not None:
        camera_cfg["fps"] = args.camera_fps
    if args.camera_backend is not None:
        camera_cfg["backend"] = args.camera_backend
    if args.camera_fourcc is not None:
        camera_cfg["fourcc"] = args.camera_fourcc
    if args.camera_buffer_size is not None:
        camera_cfg["buffer_size"] = args.camera_buffer_size
    if args.camera_startup_timeout is not None:
        camera_cfg["startup_timeout"] = args.camera_startup_timeout
    if args.camera_read_timeout is not None:
        camera_cfg["read_timeout"] = args.camera_read_timeout
    if args.jpeg_quality is not None:
        web_cfg["jpeg_quality"] = args.jpeg_quality
    if args.max_frames is not None:
        realtime_cfg["max_frames"] = args.max_frames

    run_ascend_web_demo(
        config=config,
        model_path=args.model,
        source=args.source,
        device_id=args.device_id,
        output_dtype=args.output_dtype,
        host=args.host,
        port=args.port,
        browser_voice_default=args.browser_voice_default,
        server_voice_enabled=args.server_voice,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_path = ROOT / "outputs" / "ascend_web_demo_error.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
