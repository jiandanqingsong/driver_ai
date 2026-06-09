"""EMA temporal smoothing for frame-level class probabilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EMAState:
    raw_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    raw_index: int
    raw_confidence: float
    smoothed_index: int
    smoothed_confidence: float
    margin: float
    frame_index: int


class EMASmoother:
    """Exponential moving average smoother.

    Formula:

        p_smooth[t] = alpha * p_raw[t] + (1 - alpha) * p_smooth[t - 1]

    Larger alpha reacts faster. Smaller alpha is steadier but slower.
    """

    def __init__(
        self,
        alpha: float,
        num_classes: int,
        reset_after_seconds: float | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.num_classes = num_classes
        self.reset_after_seconds = reset_after_seconds
        self.state: np.ndarray | None = None
        self.frame_index = 0
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.state = None
        self.frame_index = 0
        self.last_timestamp = None

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        """Update EMA and return only smoothed probabilities."""
        return self.update_with_state(probabilities).smoothed_probabilities

    def update_with_state(self, probabilities: np.ndarray, timestamp: float | None = None) -> EMAState:
        """Update EMA and return raw/smoothed prediction details."""
        timestamp = time.time() if timestamp is None else timestamp
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape[-1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {probabilities.shape[-1]}.")

        probabilities = self._normalize(probabilities)
        if self._should_reset(timestamp):
            self.state = None

        if self.state is None:
            self.state = probabilities.copy()
        else:
            self.state = self.alpha * probabilities + (1.0 - self.alpha) * self.state
            self.state = self._normalize(self.state)

        self.frame_index += 1
        self.last_timestamp = timestamp

        raw_index = int(np.argmax(probabilities))
        smoothed_index = int(np.argmax(self.state))
        sorted_probs = np.sort(self.state)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if sorted_probs.size >= 2 else 0.0

        return EMAState(
            raw_probabilities=probabilities.copy(),
            smoothed_probabilities=self.state.copy(),
            raw_index=raw_index,
            raw_confidence=float(probabilities[raw_index]),
            smoothed_index=smoothed_index,
            smoothed_confidence=float(self.state[smoothed_index]),
            margin=margin,
            frame_index=self.frame_index,
        )

    def _should_reset(self, timestamp: float) -> bool:
        if self.reset_after_seconds is None or self.last_timestamp is None:
            return False
        return timestamp - self.last_timestamp > self.reset_after_seconds

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        total = float(probabilities.sum())
        if total <= 1e-8:
            return np.full_like(probabilities, 1.0 / probabilities.size)
        return probabilities / total


def smooth_probability_sequence(
    probabilities: list[np.ndarray] | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Smooth an offline sequence shaped `(T, C)` for tests or analysis."""
    sequence = np.asarray(probabilities, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError("Expected probability sequence with shape (T, C).")

    smoother = EMASmoother(alpha=alpha, num_classes=sequence.shape[1])
    return np.stack([smoother.update(step) for step in sequence], axis=0)
