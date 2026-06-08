"""Dynamic risk scoring and abnormal duration tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RiskState:
    score: float
    level: str
    abnormal_seconds: float
    should_alarm: bool


class RiskAssessor:
    def __init__(
        self,
        class_risk_weights: dict[str, float],
        thresholds: dict[str, float],
        abnormal_hold_seconds: float,
        risk_decay: float = 0.92,
    ) -> None:
        self.class_risk_weights = class_risk_weights
        self.thresholds = thresholds
        self.abnormal_hold_seconds = abnormal_hold_seconds
        self.risk_decay = risk_decay
        self.score = 0.0
        self.abnormal_start_time: float | None = None

    def update(self, label: str, confidence: float, now: float | None = None) -> RiskState:
        now = time.time() if now is None else now
        base_risk = float(self.class_risk_weights.get(label, 0.0))
        confidence = max(0.0, min(1.0, float(confidence)))
        instant_risk = base_risk * confidence
        self.score = self.score * self.risk_decay + instant_risk * (1.0 - self.risk_decay)
        self.score = max(0.0, min(100.0, self.score))

        is_abnormal = base_risk > 0
        if is_abnormal and self.abnormal_start_time is None:
            self.abnormal_start_time = now
        if not is_abnormal:
            self.abnormal_start_time = None

        abnormal_seconds = 0.0 if self.abnormal_start_time is None else now - self.abnormal_start_time
        level = self.level_from_score(self.score)
        should_alarm = level in {"medium", "high"} and abnormal_seconds >= self.abnormal_hold_seconds
        return RiskState(
            score=self.score,
            level=level,
            abnormal_seconds=abnormal_seconds,
            should_alarm=should_alarm,
        )

    def level_from_score(self, score: float) -> str:
        if score >= float(self.thresholds["high"]):
            return "high"
        if score >= float(self.thresholds["medium"]):
            return "medium"
        if score >= float(self.thresholds["low"]):
            return "low"
        return "normal"
