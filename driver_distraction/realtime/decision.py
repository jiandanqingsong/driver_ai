"""Confusion-aware temporal decision filtering for realtime inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DecisionState:
    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    margin: float
    is_ambiguous: bool
    stable_frames: int


class TemporalDecisionFilter:
    """Stabilize frame-level predictions before risk scoring.

    MobileNetV3-Large is lightweight but can confuse visually similar cockpit
    states. This filter uses probability margin, known confusion pairs and
    consecutive-frame confirmation to avoid reacting to a single noisy frame.
    """

    def __init__(
        self,
        class_names: list[str],
        unknown_label: str,
        confusion_pairs: list[list[str]] | None = None,
        ambiguous_margin: float = 0.12,
        switch_margin: float = 0.08,
        min_stable_frames: int = 4,
        safe_restore_frames: int = 8,
        safe_label: str = "safe_driving",
    ) -> None:
        self.class_names = class_names
        self.unknown_label = unknown_label
        self.confusion_pairs = {frozenset(pair) for pair in (confusion_pairs or [])}
        self.ambiguous_margin = ambiguous_margin
        self.switch_margin = switch_margin
        self.min_stable_frames = min_stable_frames
        self.safe_restore_frames = safe_restore_frames
        self.safe_label = safe_label
        self.stable_label: str | None = None
        self.candidate_label: str | None = None
        self.candidate_frames = 0
        self.recent_labels: deque[str] = deque(maxlen=max(min_stable_frames, safe_restore_frames))

    def reset(self) -> None:
        self.stable_label = None
        self.candidate_label = None
        self.candidate_frames = 0
        self.recent_labels.clear()

    def update(self, probabilities: np.ndarray) -> DecisionState:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        order = np.argsort(probabilities)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        raw_label = self.class_names[top_idx]
        raw_confidence = float(probabilities[top_idx])
        second_label = self.class_names[second_idx]
        margin = float(probabilities[top_idx] - probabilities[second_idx])
        is_confusable = frozenset({raw_label, second_label}) in self.confusion_pairs
        is_ambiguous = is_confusable and margin < self.ambiguous_margin

        candidate_label = self.stable_label if is_ambiguous and self.stable_label else raw_label
        self._update_candidate(candidate_label)

        required_frames = self.safe_restore_frames if candidate_label == self.safe_label else self.min_stable_frames
        should_switch = self.candidate_frames >= required_frames
        if self.stable_label is None:
            should_switch = self.candidate_frames >= (required_frames if is_ambiguous else 1)
        elif candidate_label != self.stable_label:
            stable_idx = self.class_names.index(self.stable_label)
            confidence_gap = raw_confidence - float(probabilities[stable_idx])
            should_switch = should_switch and (confidence_gap >= self.switch_margin or not is_confusable)

        if should_switch:
            self.stable_label = candidate_label

        label = self.stable_label or self.unknown_label
        if label == self.unknown_label:
            confidence = raw_confidence
        elif label == raw_label:
            confidence = raw_confidence
        else:
            confidence = float(probabilities[self.class_names.index(label)])
        return DecisionState(
            label=label,
            confidence=confidence,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            margin=margin,
            is_ambiguous=is_ambiguous,
            stable_frames=self.candidate_frames,
        )

    def _update_candidate(self, label: str) -> None:
        if label == self.candidate_label:
            self.candidate_frames += 1
        else:
            self.candidate_label = label
            self.candidate_frames = 1
        self.recent_labels.append(label)
