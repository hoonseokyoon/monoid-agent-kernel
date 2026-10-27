"""Process-local deadlines shared by run, model and tool wait owners."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

# A private clock seam keeps deadline tests independent of asyncio's own timer.
monotonic = time.monotonic


@dataclass(frozen=True)
class _MonotonicDeadline:
    """Carry an already converted cutoff across a public epoch-input boundary.

    A carrier with ``cutoff_s=None`` explicitly owns an unbounded deadline. Its
    presence prevents a repair call from sampling wall time again.
    """

    cutoff_s: float | None

    @classmethod
    def from_epoch(cls, deadline: float | None) -> _MonotonicDeadline:
        if deadline is None:
            return cls(None)
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise ValueError("deadline must be an epoch number or None")
        try:
            epoch = float(deadline)
        except (OverflowError, ValueError) as exc:
            raise ValueError("deadline must be representable as an epoch number") from exc
        if math.isnan(epoch):
            raise ValueError("deadline must not be NaN")
        if epoch == math.inf:
            return cls(None)
        # Sample monotonic first so time spent converting never replenishes a budget.
        now = monotonic()
        return cls(now + max(0.0, epoch - time.time()))
