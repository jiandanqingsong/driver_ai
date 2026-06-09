"""Voice alarm and cooldown management."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class AlarmEvent:
    triggered: bool
    message: str
    cooldown_remaining: float


class AlarmManager:
    def __init__(
        self,
        cooldown_seconds: float,
        voice_enabled: bool = True,
        rate: int = 180,
        async_voice: bool = True,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.voice_enabled = voice_enabled
        self.async_voice = async_voice
        self.last_alarm_time = 0.0
        self.last_message = ""
        self._lock = threading.Lock()
        self._speaking = False
        self.engine = None

        if voice_enabled:
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", rate)
            except Exception as exc:
                print(f"Voice alarm disabled: {exc}")
                self.voice_enabled = False

    def can_alarm(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now - self.last_alarm_time >= self.cooldown_seconds

    def cooldown_remaining(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, self.cooldown_seconds - (now - self.last_alarm_time))

    def trigger(self, message: str, now: float | None = None) -> bool:
        return self.trigger_event(message, now).triggered

    def trigger_event(self, message: str, now: float | None = None) -> AlarmEvent:
        now = time.time() if now is None else now
        if not self.can_alarm(now):
            return AlarmEvent(False, message, self.cooldown_remaining(now))

        self.last_alarm_time = now
        self.last_message = message
        print(f"[ALARM] {message}")
        if self.voice_enabled and self.engine is not None:
            if self.async_voice:
                self._speak_async(message)
            else:
                self._speak(message)
        return AlarmEvent(True, message, self.cooldown_remaining(now))

    def _speak_async(self, message: str) -> None:
        with self._lock:
            if self._speaking:
                return
            self._speaking = True

        thread = threading.Thread(target=self._speak_worker, args=(message,), daemon=True)
        thread.start()

    def _speak_worker(self, message: str) -> None:
        try:
            self._speak(message)
        finally:
            with self._lock:
                self._speaking = False

    def _speak(self, message: str) -> None:
        if self.engine is None:
            return
        self.engine.say(message)
        self.engine.runAndWait()
