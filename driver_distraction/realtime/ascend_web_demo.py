"""Ascend OM realtime camera worker with the shared web dashboard."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from driver_distraction.constants import STATE_FARM_CLASS_NAMES
from driver_distraction.deploy.acl_infer import AscendOMClassifier, preprocess_bgr_frame, softmax
from driver_distraction.realtime.alarm import AlarmManager
from driver_distraction.realtime.capture import open_optimized_capture
from driver_distraction.realtime.decision import TemporalDecisionFilter
from driver_distraction.realtime.risk import RiskAssessor
from driver_distraction.realtime.smoothing import EMASmoother
from driver_distraction.realtime.web_server import build_index_html, create_web_server, safe_print


class FPSMeter:
    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(alpha)
        self.last_tick = time.perf_counter()
        self.fps: float | None = None

    def update(self) -> float:
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_tick, 1e-8)
        self.last_tick = now
        if self.fps is None:
            self.fps = instant_fps
        else:
            self.fps = self.alpha * self.fps + (1.0 - self.alpha) * instant_fps
        return self.fps


class AscendRealtimeWebWorker:
    """Own the camera, OM model and temporal risk state for the board web demo."""

    def __init__(
        self,
        config: dict[str, Any],
        model_path: str | Path,
        source: int | str | None = None,
        device_id: int = 0,
        output_dtype: str = "auto",
        server_voice_enabled: bool = False,
        acl_module: ModuleType | None = None,
    ) -> None:
        self.config = config
        self.realtime_cfg = config["realtime"]
        self.source = self.realtime_cfg["source"] if source is None else source
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        self.device_id = int(device_id)
        self.model_path = Path(model_path)
        self.class_names = list(config["data"].get("class_names", STATE_FARM_CLASS_NAMES))
        self.num_classes = len(self.class_names)
        self.unknown_label = str(self.realtime_cfg["unknown_label"])
        self.input_size = int(self.realtime_cfg.get("input_size", 224))
        self.resize_size = int(config.get("data", {}).get("augmentation", {}).get("resize_size", 256))
        self.jpeg_quality = int(self.realtime_cfg.get("web", {}).get("jpeg_quality", 85))
        self.max_frames = self.realtime_cfg.get("max_frames")
        self.max_frames = int(self.max_frames) if self.max_frames is not None else None
        self.server_voice_enabled = bool(server_voice_enabled)

        self.model = AscendOMClassifier(
            model_path=self.model_path,
            device_id=self.device_id,
            num_classes=self.num_classes,
            output_dtype=output_dtype,
            acl_module=acl_module,
        )

        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._closed = False
        self._last_alarm_id = 0
        self._last_alarm_message = ""
        self._last_alarm_time = 0.0
        self._last_alarm_label = ""
        self._last_alarm_level = "normal"
        self._status = self._initial_stats("initializing")
        self._build_runtime_state()

    def _build_runtime_state(self) -> None:
        smoothing_cfg = self.realtime_cfg.get("temporal_smoothing", {})
        self.smoothing_enabled = bool(smoothing_cfg.get("enabled", True))
        self.smoother = EMASmoother(
            alpha=float(smoothing_cfg.get("alpha", self.realtime_cfg.get("ema_alpha", 0.35))),
            num_classes=self.num_classes,
            reset_after_seconds=smoothing_cfg.get("reset_after_seconds"),
        )
        self.risk_assessor = RiskAssessor(
            class_risk_weights=dict(self.realtime_cfg["class_risk_weights"]),
            thresholds=dict(self.realtime_cfg["risk_thresholds"]),
            abnormal_hold_seconds=float(self.realtime_cfg["abnormal_hold_seconds"]),
            risk_decay=float(self.realtime_cfg["risk_decay"]),
        )

        alarm_cfg = self.realtime_cfg.get("voice", {})
        self.alarm = AlarmManager(
            cooldown_seconds=float(self.realtime_cfg["alarm_cooldown_seconds"]),
            voice_enabled=self.server_voice_enabled,
            rate=int(alarm_cfg.get("rate", 180)),
            async_voice=bool(alarm_cfg.get("async", True)),
        )

        decision_cfg = self.realtime_cfg.get("decision_filter", {})
        self.decision_filter = None
        if bool(decision_cfg.get("enabled", True)):
            self.decision_filter = TemporalDecisionFilter(
                class_names=self.class_names,
                unknown_label=self.unknown_label,
                confusion_pairs=decision_cfg.get("confusion_pairs", []),
                ambiguous_margin=float(decision_cfg.get("ambiguous_margin", 0.12)),
                switch_margin=float(decision_cfg.get("switch_margin", 0.08)),
                min_stable_frames=int(decision_cfg.get("min_stable_frames", 4)),
                safe_restore_frames=int(decision_cfg.get("safe_restore_frames", 8)),
                safe_label=str(decision_cfg.get("safe_label", "safe_driving")),
            )

        self.fps_meter = FPSMeter()
        self.started_at = time.time()
        self.frame_count = 0
        self.alarm_count = 0
        self.class_counts = {name: 0 for name in [*self.class_names, self.unknown_label]}
        self.risk_level_counts = {name: 0 for name in ("normal", "low", "medium", "high")}

    def _initial_stats(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "source": str(self.source),
            "device": f"Ascend:{self.device_id}",
            "model": "mobilenet_v3_large",
            "checkpoint": str(self.model_path),
            "frame_count": 0,
            "runtime_seconds": 0.0,
            "fps": 0.0,
            "label": "waiting",
            "confidence": 0.0,
            "raw_label": "waiting",
            "raw_confidence": 0.0,
            "margin": 0.0,
            "is_ambiguous": False,
            "risk": {
                "score": 0.0,
                "level": "normal",
                "abnormal_seconds": 0.0,
                "is_abnormal": False,
                "abnormal_label": None,
            },
            "cooldown_remaining": 0.0,
            "top_predictions": [],
            "class_counts": {name: 0 for name in [*self.class_names, self.unknown_label]},
            "risk_level_counts": {name: 0 for name in ("normal", "low", "medium", "high")},
            "alarm": {
                "id": self._last_alarm_id,
                "count": 0,
                "message": "",
                "last_time": 0.0,
                "label": self._last_alarm_label,
                "level": self._last_alarm_level,
                "triggered": False,
            },
            "config": {
                "smoothing_enabled": bool(
                    self.realtime_cfg.get("temporal_smoothing", {}).get("enabled", True)
                ),
                "decision_filter_enabled": bool(
                    self.realtime_cfg.get("decision_filter", {}).get("enabled", True)
                ),
                "confidence_threshold": float(self.realtime_cfg["confidence_threshold"]),
                "abnormal_hold_seconds": float(self.realtime_cfg["abnormal_hold_seconds"]),
                "alarm_cooldown_seconds": float(self.realtime_cfg["alarm_cooldown_seconds"]),
            },
            "error": "",
            "updated_at": time.time(),
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ascend-web-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop()
        self.model.close()

    def reset_statistics(self) -> None:
        with self._condition:
            self._build_runtime_state()
            self._last_alarm_id = 0
            self._last_alarm_message = ""
            self._last_alarm_time = 0.0
            self._last_alarm_label = ""
            self._last_alarm_level = "normal"
            current = self._status.copy()
            current.update(
                {
                    "frame_count": 0,
                    "runtime_seconds": 0.0,
                    "class_counts": self.class_counts.copy(),
                    "risk_level_counts": self.risk_level_counts.copy(),
                    "alarm": {
                        "id": 0,
                        "count": 0,
                        "message": "",
                        "last_time": 0.0,
                        "label": "",
                        "level": "normal",
                        "triggered": False,
                    },
                    "updated_at": time.time(),
                }
            )
            self._status = current
            self._condition.notify_all()

    def get_latest_jpeg(self, timeout: float = 2.0) -> bytes | None:
        with self._condition:
            if self._latest_jpeg is None:
                self._condition.wait(timeout=timeout)
            return self._latest_jpeg

    def get_stats(self) -> dict[str, Any]:
        with self._condition:
            return json.loads(json.dumps(self._status))

    def _run(self) -> None:
        capture = None
        try:
            self._set_status("opening")
            camera_cfg = self.realtime_cfg.get("camera", {})
            capture = open_optimized_capture(
                source=self.source,
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

            self._set_status("running")
            while not self._stop_event.is_set():
                if self.max_frames is not None and self.frame_count >= self.max_frames:
                    self._set_status("finished")
                    break

                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("finished")
                    break
                self._process_frame(frame)
        except Exception as exc:
            self._set_status("error", error=str(exc))
        finally:
            if capture is not None:
                capture.release()
            if self._stop_event.is_set():
                self._set_status("stopped")

    def _process_frame(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        tensor = preprocess_bgr_frame(
            frame,
            input_size=self.input_size,
            resize_size=self.resize_size,
        )
        logits = self.model.infer(tensor)
        if logits.size < self.num_classes:
            raise ValueError(
                f"OM output contains {logits.size} values, expected at least {self.num_classes}"
            )
        probabilities = softmax(logits[: self.num_classes])

        if self.smoothing_enabled:
            ema_state = self.smoother.update_with_state(probabilities)
            smoothed = ema_state.smoothed_probabilities
        else:
            ema_state = None
            smoothed = probabilities

        if self.decision_filter is not None:
            decision = self.decision_filter.update(smoothed)
            label = decision.label
            confidence = decision.confidence
            raw_label = self.class_names[ema_state.raw_index] if ema_state is not None else decision.raw_label
            raw_confidence = (
                float(ema_state.raw_confidence) if ema_state is not None else decision.raw_confidence
            )
            margin = float(ema_state.margin) if ema_state is not None else decision.margin
            is_ambiguous = bool(decision.is_ambiguous)
        else:
            prediction_index = int(np.argmax(smoothed))
            confidence = float(smoothed[prediction_index])
            label = self.class_names[prediction_index]
            raw_label = self.class_names[ema_state.raw_index] if ema_state is not None else label
            raw_confidence = (
                float(ema_state.raw_confidence) if ema_state is not None else confidence
            )
            margin = float(ema_state.margin) if ema_state is not None else 0.0
            is_ambiguous = False

        if confidence < float(self.realtime_cfg["confidence_threshold"]):
            label = self.unknown_label

        risk_state = self.risk_assessor.update(label, confidence)
        alarm_event = None
        if risk_state.should_alarm:
            alarm_cfg = self.realtime_cfg.get("voice", {})
            template = str(
                alarm_cfg.get(
                    "message_template",
                    "Warning, distracted driving detected: {label}, risk level {level}",
                )
            )
            message = template.format(
                label=label,
                level=risk_state.level,
                score=risk_state.score,
            )
            alarm_event = self.alarm.trigger_event(message)
            if alarm_event.triggered:
                self.alarm_count += 1
                self._last_alarm_id += 1
                self._last_alarm_message = message
                self._last_alarm_time = time.time()
                self._last_alarm_label = label
                self._last_alarm_level = risk_state.level

        fps = self.fps_meter.update()
        cooldown_remaining = self.alarm.cooldown_remaining()
        self.class_counts[label] = self.class_counts.get(label, 0) + 1
        self.risk_level_counts[risk_state.level] = (
            self.risk_level_counts.get(risk_state.level, 0) + 1
        )

        display_frame = self._prepare_web_frame(frame)
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            display_frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            return

        alarm_triggered = bool(alarm_event.triggered) if alarm_event is not None else False
        status = {
            "status": "running",
            "source": str(self.source),
            "device": f"Ascend:{self.device_id}",
            "model": "mobilenet_v3_large",
            "checkpoint": str(self.model_path),
            "frame_count": int(self.frame_count),
            "runtime_seconds": float(time.time() - self.started_at),
            "fps": float(fps),
            "label": label,
            "confidence": float(confidence),
            "raw_label": raw_label,
            "raw_confidence": float(raw_confidence),
            "margin": float(margin),
            "is_ambiguous": is_ambiguous,
            "risk": {
                "score": float(risk_state.score),
                "level": risk_state.level,
                "abnormal_seconds": float(risk_state.abnormal_seconds),
                "is_abnormal": bool(risk_state.is_abnormal),
                "abnormal_label": risk_state.abnormal_label,
            },
            "cooldown_remaining": float(cooldown_remaining),
            "top_predictions": self._top_predictions(smoothed),
            "class_counts": self.class_counts.copy(),
            "risk_level_counts": self.risk_level_counts.copy(),
            "alarm": {
                "id": self._last_alarm_id,
                "count": int(self.alarm_count),
                "message": self._last_alarm_message,
                "last_time": float(self._last_alarm_time),
                "label": self._last_alarm_label,
                "level": self._last_alarm_level,
                "triggered": alarm_triggered,
            },
            "config": self._initial_stats("running")["config"],
            "error": "",
            "updated_at": time.time(),
        }

        with self._condition:
            self._latest_jpeg = encoded.tobytes()
            self._status = status
            self._condition.notify_all()

    def _prepare_web_frame(self, frame: np.ndarray) -> np.ndarray:
        target_width = int(self.realtime_cfg.get("display_width", 0) or 0)
        if target_width <= 0 or frame.shape[1] == target_width:
            return frame
        scale = target_width / frame.shape[1]
        target_height = max(1, int(round(frame.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

    def _top_predictions(
        self,
        probabilities: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        order = np.argsort(probabilities)[::-1][:top_k]
        return [
            {
                "label": self.class_names[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in order
        ]

    def _set_status(self, status: str, error: str = "") -> None:
        with self._condition:
            current = self._status.copy()
            current["status"] = status
            current["error"] = error
            current["updated_at"] = time.time()
            self._status = current
            self._condition.notify_all()

    def __enter__(self) -> "AscendRealtimeWebWorker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def run_ascend_web_demo(
    config: dict[str, Any],
    model_path: str | Path,
    source: int | str | None = None,
    device_id: int = 0,
    output_dtype: str = "auto",
    host: str = "0.0.0.0",
    port: int | None = None,
    browser_voice_default: bool | None = None,
    server_voice_enabled: bool = False,
    smoke_test: bool = False,
) -> None:
    web_cfg = config["realtime"].get("web", {})
    port = int(port or web_cfg.get("port", 7860))
    if browser_voice_default is None:
        browser_voice_default = bool(web_cfg.get("browser_voice_default", False))

    worker = AscendRealtimeWebWorker(
        config=config,
        model_path=model_path,
        source=source,
        device_id=device_id,
        output_dtype=output_dtype,
        server_voice_enabled=server_voice_enabled,
    )
    worker.start()

    if smoke_test:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            stats = worker.get_stats()
            if stats.get("frame_count", 0) > 0 or stats.get("status") in {"error", "finished"}:
                print(json.dumps(stats, ensure_ascii=False, indent=2))
                worker.close()
                return
            time.sleep(0.2)
        worker.close()
        raise TimeoutError("Ascend web demo smoke test timed out before processing a frame.")

    index_html = build_index_html(
        browser_voice_default=bool(browser_voice_default),
        stats_interval_ms=int(web_cfg.get("stats_interval_ms", 500)),
    )
    server = create_web_server(host, port, worker, index_html)
    safe_print(f"Ascend OM web demo running at http://{host}:{port}")
    safe_print(f"Open http://<board-ip>:{port} from a browser on the same network.")
    safe_print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        safe_print("Stopping Ascend OM web demo.")
    finally:
        server.shutdown()
        server.server_close()
        worker.close()
