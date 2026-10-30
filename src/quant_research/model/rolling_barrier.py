"""Incremental model contract for point-in-time minute-bar forecasts.

The private implementation owns its rolling state; the public framework only
passes an initial history, subsequent completed bars, and forecast parameters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pyarrow as pa

from quant_research.model.barrier import BarrierForecast, BarrierModelInput


class RollingBarrierModel(Protocol):
    """Stateful, per-security model supporting efficient incremental updates.

    Initialize once with a validated five-session input. Thereafter, update
    with *new* bars only, in ascending bar-end timestamp order. No future
    data may be passed to either method. A model instance represents a single
    security; construct a separate instance for each security being traded.
    """

    def initialize(self, data: BarrierModelInput) -> None:
        """Initialize internal state from the historical minute window."""
        ...

    def update(self, new_bars: pa.Table, *, decision_at: datetime) -> None:
        """Incorporate only newly available bars, without rebuilding history."""
        ...

    def forecast(
        self,
        *,
        decision_at: datetime,
        upper_barrier_pct: float,
        lower_barrier_pct: float,
        max_holding_minutes: int,
    ) -> BarrierForecast:
        """Predict using accumulated state without recomputing the window."""
        ...
