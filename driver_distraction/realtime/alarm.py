"""Voice alarm and cooldown management."""

from __future__ import annotations

import time


class AlarmManager:
    def __init__(self, cooldown_seconds: float, voice_enabled: bool = True, rate: int = 180) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.voice_enabled = voice_enabled
        self.last_alarm_time = 0.0
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

    def trigger(self, message: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if not self.can_alarm(now):
            return False

        self.last_alarm_time = now
        print(f"[ALARM] {message}")
        if self.voice_enabled and self.engine is not None:
            self.engine.say(message)
            self.engine.runAndWait()
        return True
