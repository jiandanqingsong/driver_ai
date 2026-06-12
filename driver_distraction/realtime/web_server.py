"""Shared HTTP and MJPEG server for realtime inference workers."""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


WEB_DASHBOARD_PATH = Path(__file__).with_name("web_dashboard.html")


def build_index_html(browser_voice_default: bool = False, stats_interval_ms: int = 500) -> str:
    html = WEB_DASHBOARD_PATH.read_text(encoding="utf-8")
    return (
        html.replace("__VOICE_CHECKED__", "checked" if browser_voice_default else "")
        .replace("__STATS_INTERVAL_MS__", str(int(stats_interval_ms)))
    )


def make_handler(worker: Any, index_html: str):
    class WebDemoHandler(BaseHTTPRequestHandler):
        server_version = "DriverDistractionWebDemo/2.0"

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


def create_web_server(
    host: str,
    port: int,
    worker: Any,
    index_html: str,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), make_handler(worker, index_html))


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except Exception:
        return
