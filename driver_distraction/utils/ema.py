"""Exponential moving average (EMA) of model weights.

Keeping a slowly-updated shadow copy of the training weights usually yields a
smoother model that generalizes better than the final raw weights. After every
optimizer step the shadow tracks the live model as::

    ema = decay * ema + (1 - decay) * model

Buffers that are not floating point (e.g. ``num_batches_tracked``) are copied
verbatim rather than averaged.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, peeling a DataParallel/DDP wrapper."""
    return getattr(model, "module", model)


class ModelEMA:
    """Maintains an exponential moving average of a model's parameters/buffers.

    Args:
        model: The live training model to shadow.
        decay: Target smoothing factor in ``[0, 1)``; higher means slower / more
            stable averaging.
        device: Optional device to hold the shadow weights on. Defaults to the
            model's device.
        warmup: If True, ramp the effective decay up from 0 so the early average
            is not dominated by the (near-random) initial weights.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device | None = None,
        warmup: bool = True,
    ) -> None:
        self.module = copy.deepcopy(_unwrap(model)).eval()
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.device = device
        if device is not None:
            self.module.to(device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    def _current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        # timm-style ramp: small at first, asymptotically approaching `decay`.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self._current_decay()
        model_state = _unwrap(model).state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = model_state[key].detach().to(ema_value.device)
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "module": self.module.state_dict(),
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "module" in state:
            self.module.load_state_dict(state["module"])
            self.num_updates = int(state.get("num_updates", 0))
            self.decay = float(state.get("decay", self.decay))
        else:
            # Backwards-compatible with a bare module state dict.
            self.module.load_state_dict(state)
