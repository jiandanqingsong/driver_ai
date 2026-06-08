"""Realtime camera demo with smoothing, risk scoring and visual alerts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from driver_distraction.constants import STATE_FARM_CLASS_NAMES
from driver_distraction.data.transforms import build_realtime_transform
from driver_distraction.models.factory import build_model
from driver_distraction.realtime.alarm import AlarmManager
from driver_distraction.realtime.risk import RiskAssessor
from driver_distraction.realtime.smoothing import EMASmoother
from driver_distraction.utils.checkpoint import load_checkpoint


def load_realtime_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    realtime_cfg = config["realtime"]
    model = build_model(
        model_name=str(realtime_cfg.get("model_name", config["train"]["model_name"])),
        num_classes=int(config["data"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = load_checkpoint(realtime_cfg["checkpoint"], map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_frame(model, frame_bgr: np.ndarray, transform, device: torch.device) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs


def draw_dashboard(frame: np.ndarray, label: str, confidence: float, risk_state, display_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width != display_width:
        scale = display_width / width
        frame = cv2.resize(frame, (display_width, int(height * scale)))

    level_colors = {
        "normal": (40, 180, 80),
        "low": (0, 210, 255),
        "medium": (0, 160, 255),
        "high": (0, 0, 255),
    }
    color = level_colors.get(risk_state.level, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (18, 18, 18), -1)
    cv2.putText(frame, f"Behavior: {label} ({confidence:.2f})", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        frame,
        f"Risk: {risk_state.level.upper()} score={risk_state.score:.1f} hold={risk_state.abnormal_seconds:.1f}s",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    bar_x, bar_y, bar_w, bar_h = 20, frame.shape[0] - 36, frame.shape[1] - 40, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    fill_w = int(bar_w * min(max(risk_state.score, 0.0), 100.0) / 100.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    return frame


def run_camera_demo(config: dict[str, Any], source: int | str | None = None) -> None:
    realtime_cfg = config["realtime"]
    device_name = config["project"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    model = load_realtime_model(config, device)
    transform = build_realtime_transform(int(realtime_cfg["input_size"]))
    smoother = EMASmoother(float(realtime_cfg["ema_alpha"]), int(config["data"]["num_classes"]))
    risk_assessor = RiskAssessor(
        class_risk_weights=dict(realtime_cfg["class_risk_weights"]),
        thresholds=dict(realtime_cfg["risk_thresholds"]),
        abnormal_hold_seconds=float(realtime_cfg["abnormal_hold_seconds"]),
        risk_decay=float(realtime_cfg["risk_decay"]),
    )
    alarm_cfg = realtime_cfg.get("voice", {})
    alarm = AlarmManager(
        cooldown_seconds=float(realtime_cfg["alarm_cooldown_seconds"]),
        voice_enabled=bool(alarm_cfg.get("enabled", True)),
        rate=int(alarm_cfg.get("rate", 180)),
    )

    video_source = realtime_cfg["source"] if source is None else source
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    class_names = list(config["data"].get("class_names", STATE_FARM_CLASS_NAMES))
    unknown_label = str(realtime_cfg["unknown_label"])
    confidence_threshold = float(realtime_cfg["confidence_threshold"])
    message_template = str(alarm_cfg.get("message_template", "Distracted driving detected: {label}"))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            probs = predict_frame(model, frame, transform, device)
            smoothed = smoother.update(probs)
            pred_idx = int(np.argmax(smoothed))
            confidence = float(smoothed[pred_idx])
            label = class_names[pred_idx]
            if confidence < confidence_threshold:
                label = unknown_label

            risk_state = risk_assessor.update(label, confidence)
            if risk_state.should_alarm:
                message = message_template.format(label=label, level=risk_state.level, score=risk_state.score)
                alarm.trigger(message)

            frame = draw_dashboard(frame, label, confidence, risk_state, int(realtime_cfg["display_width"]))
            cv2.imshow("Driver Distraction Risk Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
