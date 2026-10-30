"""Contracts for private intraday barrier prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pyarrow as pa

from quant_research.core import SecurityId


@dataclass(frozen=True)
class BarrierModelInput:
    """Point-in-time minute OHLCV data for one security."""

    security_id: SecurityId
    decision_at: datetime
    minute_bars: pa.Table

    upper_barrier_pct: float
    lower_barrier_pct: float
    max_holding_minutes: int


@dataclass(frozen=True)
class BarrierForecast:
    """Predicted first-barrier outcome probabilities."""

    security_id: SecurityId
    generated_at: datetime

    probability_upper_first: float
    probability_lower_first: float
    probability_neither: float


class BarrierModel(Protocol):
    """Interface implemented by a private prediction model."""

    def predict(
        self,
        data: BarrierModelInput,
    ) -> BarrierForecast: ...
