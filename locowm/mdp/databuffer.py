from __future__ import annotations

import math
import torch


class StepwiseLPF:
    """Cache one EMA update per environment step across all observation/reward reads.

    Use raw values for newly reset environments so previous-episode history
    does not affect the first frame."""

    def __init__(self, cut_off_frequency: float, control_frequency: float):
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * cut_off_frequency / control_frequency)
        self._y: torch.Tensor | None = None  # Filtered output cached for the current step.
        self._last_step: int | None = None  # Step index of the last filter update.

    def __call__(
        self,
        x: torch.Tensor,
        step: int,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Initialize from the raw value on first use or a batch-shape change.
        if self._y is None or self._y.shape != x.shape:
            self._y = x.clone()
            self._last_step = None

        # Reuse the cached value for repeated reads within a step.
        if step == self._last_step:
            return self._y

        y = self.alpha * x + (1.0 - self.alpha) * self._y
        if reset_mask is not None:
            y = torch.where(reset_mask, x, y)  # Discard old filter history at the start of each episode.

        self._y = y
        self._last_step = step
        return y
