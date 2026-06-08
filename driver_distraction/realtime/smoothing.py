"""Temporal probability smoothing."""

from __future__ import annotations

import numpy as np


class EMASmoother:
    def __init__(self, alpha: float, num_classes: int) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.num_classes = num_classes
        self.state: np.ndarray | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape[-1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {probabilities.shape[-1]}.")
        if self.state is None:
            self.state = probabilities
        else:
            self.state = self.alpha * probabilities + (1.0 - self.alpha) * self.state
        self.state = self.state / max(float(self.state.sum()), 1e-8)
        return self.state
