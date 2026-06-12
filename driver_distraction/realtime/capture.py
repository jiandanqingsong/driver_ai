"""Optimized OpenCV capture setup for cameras and media files."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BACKEND_CODES = {
    "default": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
}


@dataclass(frozen=True)
class CaptureInfo:
    source: int | str
    backend: str
    width: int
    height: int
    fps: float
    fourcc: str
    open_seconds: float
    threaded: bool


def normalize_video_source(source: int | str) -> int | str:
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def backend_candidates(source: int | str, backend: str) -> list[str]:
    if not isinstance(source, int):
        return ["default"]

    backend = backend.strip().lower()
    if backend != "auto":
        if backend not in BACKEND_CODES:
            choices = ", ".join(["auto", *BACKEND_CODES])
            raise ValueError(f"Unsupported camera backend: {backend}. Choose one of: {choices}.")
        return [backend]

    if os.name == "nt":
        # MSMF can spend over a minute negotiating some USB cameras and may
        # ignore OpenCV's timeout properties. Keep auto startup deterministic;
        # users can still request MSMF explicitly when a camera requires it.
        return ["dshow"]
    if os.name == "posix":
        return ["v4l2", "default"]
    return ["default"]


def decode_fourcc(value: float) -> str:
    integer = int(value)
    if integer <= 0:
        return ""
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


class OptimizedVideoCapture:
    """Open a camera with deterministic backend settings and latest-frame reads.

    Live cameras are read on a background thread. Only the newest frame is kept,
    which prevents inference from processing an increasingly stale OpenCV buffer.
    Files and image sources retain normal sequential reads.
    """

    def __init__(
        self,
        source: int | str,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        backend: str = "auto",
        fourcc: str = "MJPG",
        buffer_size: int = 1,
        startup_timeout: float = 8.0,
        read_timeout: float = 2.0,
        threaded: bool = True,
    ) -> None:
        self.source = normalize_video_source(source)
        self.is_live_camera = isinstance(self.source, int)
        self.read_timeout = max(0.1, float(read_timeout))
        self.threaded = bool(threaded and self.is_live_camera)
        self._cap: cv2.VideoCapture | None = None
        self._first_frame: np.ndarray | None = None
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._consumer_sequence = 0
        self._ended = False
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._reader_thread: threading.Thread | None = None

        started_at = time.perf_counter()
        errors: list[str] = []
        for candidate in backend_candidates(self.source, backend):
            cap = self._open_candidate(
                candidate,
                width=width,
                height=height,
                fps=fps,
                fourcc=fourcc,
                buffer_size=buffer_size,
                startup_timeout=startup_timeout,
            )
            if cap is None:
                errors.append(candidate)
                continue
            self._cap = cap
            self.backend = self._backend_name(cap, candidate)
            break

        if self._cap is None:
            attempted = ", ".join(errors) if errors else backend
            raise RuntimeError(f"Cannot open video source {self.source!r}; attempted backends: {attempted}.")

        self.info = CaptureInfo(
            source=self.source,
            backend=self.backend,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)),
            fourcc=decode_fourcc(self._cap.get(cv2.CAP_PROP_FOURCC)),
            open_seconds=time.perf_counter() - started_at,
            threaded=self.threaded,
        )

        if self.threaded:
            with self._condition:
                self._latest_frame = self._first_frame
                self._sequence = 1
            self._first_frame = None
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="opencv-camera-reader",
                daemon=True,
            )
            self._reader_thread.start()

    def _open_candidate(
        self,
        backend: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        buffer_size: int,
        startup_timeout: float,
    ) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, max(100, int(startup_timeout * 1000)))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, max(100, int(self.read_timeout * 1000)))

        opened = cap.open(self.source, BACKEND_CODES[backend])
        if not opened or not cap.isOpened():
            cap.release()
            return None

        if self.is_live_camera:
            if fourcc:
                code = cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4))
                cap.set(cv2.CAP_PROP_FOURCC, code)
            if width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(fps))
            if buffer_size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        self._first_frame = frame
        return cap

    @staticmethod
    def _backend_name(cap: cv2.VideoCapture, fallback: str) -> str:
        try:
            return str(cap.getBackendName()).lower()
        except cv2.error:
            return fallback

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.threaded:
            if self._first_frame is not None:
                frame = self._first_frame
                self._first_frame = None
                return True, frame
            if self._cap is None:
                return False, None
            return self._cap.read()

        deadline = time.monotonic() + self.read_timeout
        with self._condition:
            while self._sequence <= self._consumer_sequence and not self._ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(timeout=remaining)

            if self._latest_frame is None:
                return False, None
            self._consumer_sequence = self._sequence
            return True, self._latest_frame.copy()

    def get(self, property_id: int) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(property_id))

    def release(self) -> None:
        self._stop_event.set()
        cap = self._cap
        self._cap = None
        if cap is not None:
            cap.release()
        with self._condition:
            self._ended = True
            self._condition.notify_all()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                with self._condition:
                    self._ended = True
                    self._condition.notify_all()
                break
            with self._condition:
                self._latest_frame = frame
                self._sequence += 1
                self._condition.notify_all()


def open_optimized_capture(
    source: int | str,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
    backend: str = "auto",
    fourcc: str = "MJPG",
    buffer_size: int = 1,
    startup_timeout: float = 8.0,
    read_timeout: float = 2.0,
    threaded: bool = True,
) -> OptimizedVideoCapture:
    capture = OptimizedVideoCapture(
        source=source,
        width=width,
        height=height,
        fps=fps,
        backend=backend,
        fourcc=fourcc,
        buffer_size=buffer_size,
        startup_timeout=startup_timeout,
        read_timeout=read_timeout,
        threaded=threaded,
    )
    info = capture.info
    print(
        f"Video source ready in {info.open_seconds:.2f}s: backend={info.backend}, "
        f"size={info.width}x{info.height}, fps={info.fps:.1f}, "
        f"fourcc={info.fourcc or 'unknown'}, latest_frame={info.threaded}"
    )
    return capture
