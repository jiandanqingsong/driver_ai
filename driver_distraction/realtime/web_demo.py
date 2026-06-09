"""Web dashboard for realtime driver distraction inference."""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from driver_distraction.constants import STATE_FARM_CLASS_NAMES
from driver_distraction.data.transforms import build_realtime_transform
from driver_distraction.realtime.alarm import AlarmManager
from driver_distraction.realtime.camera_demo import FPSMeter, draw_dashboard, load_realtime_model, predict_frame
from driver_distraction.realtime.decision import TemporalDecisionFilter
from driver_distraction.realtime.risk import RiskAssessor
from driver_distraction.realtime.smoothing import EMASmoother


class RealtimeWebWorker:
    """Owns camera capture, inference state and latest web-facing results."""

    def __init__(self, config: dict[str, Any], source: int | str | None = None) -> None:
        self.config = config
        self.realtime_cfg = config["realtime"]
        self.source = self.realtime_cfg["source"] if source is None else source
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        device_name = str(config["project"].get("device", "cuda"))
        if device_name.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

        self.class_names = list(config["data"].get("class_names", STATE_FARM_CLASS_NAMES))
        self.unknown_label = str(self.realtime_cfg["unknown_label"])
        self.model = load_realtime_model(config, self.device)
        self.transform = build_realtime_transform(int(self.realtime_cfg["input_size"]))
        self.jpeg_quality = int(self.realtime_cfg.get("web", {}).get("jpeg_quality", 85))
        self.max_frames = self.realtime_cfg.get("max_frames")
        self.max_frames = int(self.max_frames) if self.max_frames is not None else None

        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._last_alarm_id = 0
        self._last_alarm_message = ""
        self._last_alarm_time = 0.0
        self._status = self._initial_stats("initializing")

        self._build_runtime_state()

    def _build_runtime_state(self) -> None:
        smoothing_cfg = self.realtime_cfg.get("temporal_smoothing", {})
        self.smoothing_enabled = bool(smoothing_cfg.get("enabled", True))
        self.smoother = EMASmoother(
            alpha=float(smoothing_cfg.get("alpha", self.realtime_cfg.get("ema_alpha", 0.35))),
            num_classes=int(self.config["data"]["num_classes"]),
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
            voice_enabled=bool(alarm_cfg.get("enabled", True)),
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
            "device": str(self.device),
            "model": str(self.realtime_cfg.get("model_name", self.config["train"]["model_name"])),
            "checkpoint": str(self.realtime_cfg["checkpoint"]),
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
                "triggered": False,
            },
            "config": {
                "smoothing_enabled": bool(self.realtime_cfg.get("temporal_smoothing", {}).get("enabled", True)),
                "decision_filter_enabled": bool(self.realtime_cfg.get("decision_filter", {}).get("enabled", True)),
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
        self._thread = threading.Thread(target=self._run, name="realtime-web-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def reset_statistics(self) -> None:
        with self._condition:
            self._build_runtime_state()
            self._last_alarm_id = 0
            self._last_alarm_message = ""
            self._last_alarm_time = 0.0
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
        cap = None
        try:
            self._set_status("opening")
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.source}")

            camera_cfg = self.realtime_cfg.get("camera", {})
            if camera_cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
            if camera_cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
            if camera_cfg.get("fps"):
                cap.set(cv2.CAP_PROP_FPS, float(camera_cfg["fps"]))

            self._set_status("running")
            while not self._stop_event.is_set():
                if self.max_frames is not None and self.frame_count >= self.max_frames:
                    self._set_status("finished")
                    break

                ok, frame = cap.read()
                if not ok:
                    self._set_status("finished")
                    break

                self._process_frame(frame)
        except Exception as exc:
            self._set_status("error", error=str(exc))
        finally:
            if cap is not None:
                cap.release()
            if self._stop_event.is_set():
                self._set_status("stopped")

    def _process_frame(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        probs = predict_frame(self.model, frame, self.transform, self.device)
        if self.smoothing_enabled:
            ema_state = self.smoother.update_with_state(probs)
            smoothed = ema_state.smoothed_probabilities
        else:
            ema_state = None
            smoothed = probs

        if self.decision_filter is not None:
            decision = self.decision_filter.update(smoothed)
            label = decision.label
            confidence = decision.confidence
            raw_label = self.class_names[ema_state.raw_index] if ema_state is not None else decision.raw_label
            raw_confidence = float(ema_state.raw_confidence) if ema_state is not None else decision.raw_confidence
            margin = float(ema_state.margin) if ema_state is not None else decision.margin
            is_ambiguous = bool(decision.is_ambiguous)
        else:
            pred_idx = int(np.argmax(smoothed))
            confidence = float(smoothed[pred_idx])
            label = self.class_names[pred_idx]
            raw_label = self.class_names[ema_state.raw_index] if ema_state is not None else label
            raw_confidence = float(ema_state.raw_confidence) if ema_state is not None else confidence
            margin = float(ema_state.margin) if ema_state is not None else 0.0
            is_ambiguous = False

        if confidence < float(self.realtime_cfg["confidence_threshold"]):
            label = self.unknown_label

        risk_state = self.risk_assessor.update(label, confidence)
        alarm_event = None
        if risk_state.should_alarm:
            alarm_cfg = self.realtime_cfg.get("voice", {})
            message_template = str(alarm_cfg.get("message_template", "Distracted driving detected: {label}"))
            message = message_template.format(label=label, level=risk_state.level, score=risk_state.score)
            alarm_event = self.alarm.trigger_event(message)
            if alarm_event.triggered:
                self.alarm_count += 1
                self._last_alarm_id += 1
                self._last_alarm_message = message
                self._last_alarm_time = time.time()

        fps = self.fps_meter.update()
        cooldown_remaining = self.alarm.cooldown_remaining()
        self.class_counts[label] = self.class_counts.get(label, 0) + 1
        self.risk_level_counts[risk_state.level] = self.risk_level_counts.get(risk_state.level, 0) + 1
        top_predictions = self._top_predictions(smoothed)

        display_frame = draw_dashboard(
            frame,
            label,
            confidence,
            risk_state,
            int(self.realtime_cfg["display_width"]),
            raw_label=raw_label,
            margin=margin,
            is_ambiguous=is_ambiguous,
            cooldown_remaining=cooldown_remaining,
            fps=fps,
        )
        ok, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        alarm_triggered = bool(alarm_event.triggered) if alarm_event is not None else False
        status = {
            "status": "running",
            "source": str(self.source),
            "device": str(self.device),
            "model": str(self.realtime_cfg.get("model_name", self.config["train"]["model_name"])),
            "checkpoint": str(self.realtime_cfg["checkpoint"]),
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
            "top_predictions": top_predictions,
            "class_counts": self.class_counts.copy(),
            "risk_level_counts": self.risk_level_counts.copy(),
            "alarm": {
                "id": self._last_alarm_id,
                "count": int(self.alarm_count),
                "message": self._last_alarm_message,
                "last_time": float(self._last_alarm_time),
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

    def _top_predictions(self, probabilities: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
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


def build_index_html(browser_voice_default: bool = False, stats_interval_ms: int = 500) -> str:
    checked = "checked" if browser_voice_default else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Driver Distraction Monitor</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111317;
      --panel: #1a1f27;
      --panel-2: #202733;
      --line: #343d4c;
      --text: #eef2f7;
      --muted: #9aa7b8;
      --green: #47c77b;
      --cyan: #4eb7d8;
      --amber: #e6a13c;
      --red: #e45b5b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei UI", "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #151922;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    .status-line {{
      display: flex;
      gap: 14px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--muted);
      display: inline-block;
      margin-right: 6px;
    }}
    .dot.running {{ background: var(--green); }}
    .dot.error {{ background: var(--red); }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 380px;
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 64px);
    }}
    .video-panel, .side-panel, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .video-panel {{
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .video-wrap {{
      background: #050608;
      flex: 1;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    #videoFeed {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .video-footer {{
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    .side-panel {{
      padding: 14px;
      overflow: auto;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 72px;
    }}
    .metric label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .metric strong {{
      display: block;
      font-size: 21px;
      line-height: 1.15;
      word-break: break-word;
    }}
    .metric.wide {{ grid-column: span 2; }}
    .risk-normal {{ color: var(--green); }}
    .risk-low {{ color: var(--cyan); }}
    .risk-medium {{ color: var(--amber); }}
    .risk-high {{ color: var(--red); }}
    .section-title {{
      font-size: 13px;
      font-weight: 650;
      color: #d8dee8;
      margin: 16px 0 8px;
    }}
    .bars {{
      display: grid;
      gap: 8px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 150px 1fr 48px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: #2a3240;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--cyan);
      width: 0;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    button, .toggle {{
      min-height: 36px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #252d39;
      color: var(--text);
      padding: 8px 11px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    .toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: default;
    }}
    input[type="checkbox"] {{
      width: 16px;
      height: 16px;
      accent-color: var(--cyan);
    }}
    .log {{
      min-height: 46px;
      padding: 10px 12px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    .counts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .count-item {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 9px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .side-panel {{ overflow: visible; }}
      .video-wrap {{ min-height: 300px; }}
      header {{ align-items: flex-start; height: auto; gap: 8px; padding: 14px 16px; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>驾驶员分心行为实时监控</h1>
    <div class="status-line">
      <span><span id="statusDot" class="dot"></span><span id="statusText">connecting</span></span>
      <span id="deviceText">device</span>
      <span id="modelText">model</span>
    </div>
  </header>
  <main>
    <section class="video-panel">
      <div class="video-wrap">
        <img id="videoFeed" src="/video_feed" alt="Realtime camera monitor" />
      </div>
      <div class="video-footer">
        <span id="sourceText">source</span>
        <div class="controls">
          <label class="toggle"><input id="voiceToggle" type="checkbox" {checked}>浏览器语音</label>
          <button id="testVoice" type="button">测试语音</button>
          <button id="resetStats" type="button">重置统计</button>
        </div>
      </div>
    </section>
    <aside class="side-panel">
      <div class="grid">
        <div class="metric wide"><label>稳定行为</label><strong id="labelText">waiting</strong></div>
        <div class="metric"><label>置信度</label><strong id="confText">0.00</strong></div>
        <div class="metric"><label>风险等级</label><strong id="riskLevelText" class="risk-normal">NORMAL</strong></div>
        <div class="metric"><label>风险分数</label><strong id="riskScoreText">0.0</strong></div>
        <div class="metric"><label>异常持续</label><strong id="holdText">0.0s</strong></div>
        <div class="metric"><label>FPS</label><strong id="fpsText">0.0</strong></div>
        <div class="metric"><label>帧数</label><strong id="frameText">0</strong></div>
        <div class="metric"><label>报警次数</label><strong id="alarmCountText">0</strong></div>
        <div class="metric wide"><label>原始预测</label><strong id="rawText">waiting</strong></div>
      </div>
      <div class="section-title">Top 预测概率</div>
      <div id="topBars" class="bars"></div>
      <div class="section-title">运行统计</div>
      <div id="counts" class="counts"></div>
      <div class="section-title">最近报警</div>
      <div id="alarmLog" class="log">暂无报警</div>
    </aside>
  </main>
  <script>
    const statsIntervalMs = {int(stats_interval_ms)};
    let lastAlarmId = 0;
    let lastStats = null;

    const $ = (id) => document.getElementById(id);
    const fmt = (value, digits = 1) => Number(value || 0).toFixed(digits);

    function setRiskClass(el, level) {{
      el.className = "";
      el.classList.add("risk-" + (level || "normal"));
    }}

    function speak(message) {{
      if (!("speechSynthesis" in window) || !$("voiceToggle").checked || !message) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.lang = "en-US";
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }}

    function updateBars(predictions) {{
      const host = $("topBars");
      host.innerHTML = "";
      (predictions || []).forEach((item) => {{
        const row = document.createElement("div");
        row.className = "bar-row";
        const name = document.createElement("span");
        name.textContent = item.label;
        const track = document.createElement("div");
        track.className = "bar-track";
        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = Math.max(0, Math.min(100, item.probability * 100)).toFixed(1) + "%";
        track.appendChild(fill);
        const value = document.createElement("span");
        value.textContent = fmt(item.probability, 2);
        row.append(name, track, value);
        host.appendChild(row);
      }});
    }}

    function updateCounts(counts) {{
      const host = $("counts");
      host.innerHTML = "";
      Object.entries(counts || {{}}).forEach(([name, count]) => {{
        const item = document.createElement("div");
        item.className = "count-item";
        const key = document.createElement("span");
        key.textContent = name;
        const value = document.createElement("strong");
        value.textContent = count;
        item.append(key, value);
        host.appendChild(item);
      }});
    }}

    async function refreshStats() {{
      try {{
        const res = await fetch("/api/stats", {{ cache: "no-store" }});
        const data = await res.json();
        lastStats = data;
        $("statusText").textContent = data.status || "unknown";
        $("statusDot").className = "dot " + (data.status || "");
        $("deviceText").textContent = data.device || "";
        $("modelText").textContent = data.model || "";
        $("sourceText").textContent = "source: " + (data.source || "");

        $("labelText").textContent = data.label || "waiting";
        $("confText").textContent = fmt(data.confidence, 2);
        $("riskLevelText").textContent = (data.risk?.level || "normal").toUpperCase();
        setRiskClass($("riskLevelText"), data.risk?.level || "normal");
        $("riskScoreText").textContent = fmt(data.risk?.score, 1);
        $("holdText").textContent = fmt(data.risk?.abnormal_seconds, 1) + "s";
        $("fpsText").textContent = fmt(data.fps, 1);
        $("frameText").textContent = data.frame_count || 0;
        $("alarmCountText").textContent = data.alarm?.count || 0;
        $("rawText").textContent = `${{data.raw_label || "waiting"}}  margin=${{fmt(data.margin, 2)}}`;

        updateBars(data.top_predictions);
        updateCounts(data.class_counts);

        if (data.error) {{
          $("alarmLog").textContent = "错误: " + data.error;
        }} else if (data.alarm?.message) {{
          const time = data.alarm.last_time ? new Date(data.alarm.last_time * 1000).toLocaleTimeString() : "";
          $("alarmLog").textContent = `[${{time}}] ${{data.alarm.message}}`;
        }}

        if (data.alarm?.id && data.alarm.id !== lastAlarmId) {{
          lastAlarmId = data.alarm.id;
          speak(data.alarm.message);
        }}
      }} catch (error) {{
        $("statusText").textContent = "disconnected";
        $("statusDot").className = "dot error";
        $("alarmLog").textContent = "连接后端失败: " + error;
      }}
    }}

    $("testVoice").addEventListener("click", () => speak("Driver distraction warning voice test."));
    $("resetStats").addEventListener("click", async () => {{
      await fetch("/api/reset", {{ method: "POST" }});
      lastAlarmId = 0;
      refreshStats();
    }});

    refreshStats();
    setInterval(refreshStats, statsIntervalMs);
  </script>
</body>
</html>"""


def make_handler(worker: RealtimeWebWorker, index_html: str):
    class WebDemoHandler(BaseHTTPRequestHandler):
        server_version = "DriverDistractionWebDemo/1.0"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(index_html)
                return
            if parsed.path == "/video_feed":
                self._stream_video()
                return
            if parsed.path == "/api/stats":
                self._send_json(worker.get_stats())
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "status": worker.get_stats().get("status")})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/reset":
                worker.reset_statistics()
                self._send_json({"ok": True})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_video(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while not worker._stop_event.is_set():
                jpeg = worker.get_latest_jpeg(timeout=2.0)
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break

    return WebDemoHandler


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except Exception:
        return


def run_web_demo(
    config: dict[str, Any],
    source: int | str | None = None,
    host: str | None = None,
    port: int | None = None,
    smoke_test: bool = False,
) -> None:
    web_cfg = config["realtime"].get("web", {})
    host = host or str(web_cfg.get("host", "127.0.0.1"))
    port = int(port or web_cfg.get("port", 7860))
    worker = RealtimeWebWorker(config, source=source)
    worker.start()

    if smoke_test:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            stats = worker.get_stats()
            if stats.get("frame_count", 0) > 0 or stats.get("status") in {"error", "finished"}:
                print(json.dumps(stats, ensure_ascii=False, indent=2))
                worker.stop()
                return
            time.sleep(0.2)
        worker.stop()
        raise TimeoutError("Web demo smoke test timed out before processing a frame.")

    index_html = build_index_html(
        browser_voice_default=bool(web_cfg.get("browser_voice_default", False)),
        stats_interval_ms=int(web_cfg.get("stats_interval_ms", 500)),
    )
    server = ThreadingHTTPServer((host, port), make_handler(worker, index_html))
    safe_print(f"Web demo running at http://{host}:{port}")
    safe_print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        safe_print("Stopping web demo.")
    finally:
        server.shutdown()
        server.server_close()
        worker.stop()
