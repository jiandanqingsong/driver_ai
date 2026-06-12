"""Realtime camera demo with smoothing, risk scoring and visual alerts."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from driver_distraction.constants import STATE_FARM_CLASS_NAMES
from driver_distraction.data.transforms import build_realtime_transform
from driver_distraction.models.factory import build_model
from driver_distraction.realtime.alarm import AlarmManager
from driver_distraction.realtime.capture import open_optimized_capture
from driver_distraction.realtime.decision import TemporalDecisionFilter
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


def draw_dashboard(
    frame: np.ndarray,
    label: str,
    confidence: float,
    risk_state,
    display_width: int,
    raw_label: str | None = None,
    margin: float | None = None,
    is_ambiguous: bool = False,
    cooldown_remaining: float = 0.0,
    fps: float | None = None,
) -> np.ndarray:
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
    if raw_label is not None:
        suffix = " ambiguous" if is_ambiguous else ""
        cv2.putText(
            frame,
            f"Raw: {raw_label} margin={margin:.2f}{suffix}",
            (520, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
        )
    cv2.putText(
        frame,
        f"Risk: {risk_state.level.upper()} score={risk_state.score:.1f} "
        f"hold={risk_state.abnormal_seconds:.1f}s cooldown={cooldown_remaining:.1f}s",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 130, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
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

    model_started_at = time.perf_counter()
    print(f"Loading realtime model on {device}...")
    model = load_realtime_model(config, device)
    print(f"Realtime model ready in {time.perf_counter() - model_started_at:.2f}s.")
    transform = build_realtime_transform(int(realtime_cfg["input_size"]))
    smoothing_cfg = realtime_cfg.get("temporal_smoothing", {})
    smoothing_enabled = bool(smoothing_cfg.get("enabled", True))
    smoother = EMASmoother(
        alpha=float(smoothing_cfg.get("alpha", realtime_cfg.get("ema_alpha", 0.35))),
        num_classes=int(config["data"]["num_classes"]),
        reset_after_seconds=smoothing_cfg.get("reset_after_seconds"),
    )
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
        async_voice=bool(alarm_cfg.get("async", True)),
    )

    video_source = realtime_cfg["source"] if source is None else source
    camera_cfg = realtime_cfg.get("camera", {})
    cap = open_optimized_capture(
        source=video_source,
        width=int(camera_cfg.get("width", 0) or 0),
        height=int(camera_cfg.get("height", 0) or 0),
        fps=float(camera_cfg.get("fps", 0) or 0),
        backend=str(camera_cfg.get("backend", "auto")),
        fourcc=str(camera_cfg.get("fourcc", "MJPG")),
        buffer_size=int(camera_cfg.get("buffer_size", 1)),
        startup_timeout=float(camera_cfg.get("startup_timeout", 8.0)),
        read_timeout=float(camera_cfg.get("read_timeout", 2.0)),
        threaded=bool(camera_cfg.get("threaded", True)),
    )

    window_name = str(realtime_cfg.get("window_name", "Driver Distraction Risk Demo"))
    show_window = bool(realtime_cfg.get("show_window", True))
    save_path = realtime_cfg.get("save_video_path")
    max_frames = realtime_cfg.get("max_frames")
    max_frames = int(max_frames) if max_frames is not None else None
    writer = None

    class_names = list(config["data"].get("class_names", STATE_FARM_CLASS_NAMES))
    unknown_label = str(realtime_cfg["unknown_label"])
    confidence_threshold = float(realtime_cfg["confidence_threshold"])
    message_template = str(alarm_cfg.get("message_template", "Distracted driving detected: {label}"))
    decision_cfg = realtime_cfg.get("decision_filter", {})
    decision_filter = None
    if bool(decision_cfg.get("enabled", True)):
        decision_filter = TemporalDecisionFilter(
            class_names=class_names,
            unknown_label=unknown_label,
            confusion_pairs=decision_cfg.get("confusion_pairs", []),
            ambiguous_margin=float(decision_cfg.get("ambiguous_margin", 0.12)),
            switch_margin=float(decision_cfg.get("switch_margin", 0.08)),
            min_stable_frames=int(decision_cfg.get("min_stable_frames", 4)),
            safe_restore_frames=int(decision_cfg.get("safe_restore_frames", 8)),
            safe_label=str(decision_cfg.get("safe_label", "safe_driving")),
        )

    fps_meter = FPSMeter()
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            probs = predict_frame(model, frame, transform, device)
            if smoothing_enabled:
                ema_state = smoother.update_with_state(probs)
                smoothed = ema_state.smoothed_probabilities
            else:
                ema_state = None
                smoothed = probs

            if decision_filter is not None:
                decision = decision_filter.update(smoothed)
                label = decision.label
                confidence = decision.confidence
                raw_label = class_names[ema_state.raw_index] if ema_state is not None else decision.raw_label
                margin = ema_state.margin if ema_state is not None else decision.margin
                is_ambiguous = decision.is_ambiguous
            else:
                pred_idx = int(np.argmax(smoothed))
                confidence = float(smoothed[pred_idx])
                label = class_names[pred_idx]
                raw_label = class_names[ema_state.raw_index] if ema_state is not None else label
                margin = ema_state.margin if ema_state is not None else 0.0
                is_ambiguous = False

            if confidence < confidence_threshold:
                label = unknown_label

            risk_state = risk_assessor.update(label, confidence)
            if risk_state.should_alarm:
                message = message_template.format(label=label, level=risk_state.level, score=risk_state.score)
                alarm.trigger_event(message)

            fps = fps_meter.update()
            cooldown_remaining = alarm.cooldown_remaining()

            frame = draw_dashboard(
                frame,
                label,
                confidence,
                risk_state,
                int(realtime_cfg["display_width"]),
                raw_label=raw_label,
                margin=margin,
                is_ambiguous=is_ambiguous,
                cooldown_remaining=cooldown_remaining,
                fps=fps,
            )

            if writer is None and save_path:
                writer = build_video_writer(save_path, frame, cap)
            if writer is not None:
                writer.write(frame)

            if show_window:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


class FPSMeter:
    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self.last_tick = cv2.getTickCount()
        self.fps: float | None = None

    def update(self) -> float:
        now = cv2.getTickCount()
        elapsed = (now - self.last_tick) / cv2.getTickFrequency()
        self.last_tick = now
        instant_fps = 1.0 / max(elapsed, 1e-8)
        if self.fps is None:
            self.fps = instant_fps
        else:
            self.fps = self.alpha * self.fps + (1.0 - self.alpha) * instant_fps
        return self.fps


def build_video_writer(save_path: str, frame: np.ndarray, cap) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0
    height, width = frame.shape[:2]
    return cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
