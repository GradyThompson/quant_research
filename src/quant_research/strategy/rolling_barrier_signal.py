"""Stream newly available minute bars through a private rolling model.

This module does not read files, discover model implementations, or place orders.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

import pyarrow as pa

from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.model.barrier_runner import run_barrier_model
from quant_research.model.rolling_barrier import RollingBarrierModel
from quant_research.strategy.barrier_decision import (
    BarrierDecisionConfig,
    decide_barrier_entry,
)
from quant_research.strategy.barrier_signal import BarrierSignalResult


def _utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must contain UTC-aware datetimes")
    return value


def stream_rolling_barrier_signals(
    *,
    model: RollingBarrierModel,
    initial: BarrierModelInput,
    new_bars: pa.Table,
    config: BarrierDecisionConfig,
    has_open_position: bool = False,
    has_pending_entry: bool = False,
) -> Iterator[BarrierSignalResult]:
    """Initialize once, then forecast after each *available* incoming bar.

    `new_bars` must already be in chronological availability order, for one
    security, and disjoint from initial history. Callers must supply only bars
    whose publication timestamps are known. This stream does not execute trades;
    exposure flags are constant throughout a run.
    """
    start = _utc(initial.decision_at, field="initial decision_at")
    if initial.minute_bars.num_rows == 0:
        raise ValueError("Initial history cannot be empty")
    required = {"security_id", "timestamp", "available_at"}
    if not required.issubset(new_bars.column_names):
        raise ValueError("New bars are missing required columns")

    # Validate before changing any model state, preventing partial consumption.
    previous_available = start
    previous_stamp: datetime | None = None
    events: list[datetime] = []
    for row in new_bars.select(["security_id", "timestamp", "available_at"]).to_pylist():
        stamp = _utc(row["timestamp"], field="timestamp")
        available = _utc(row["available_at"], field="available_at")
        if row["security_id"] != str(initial.security_id):
            raise ValueError("New bars contain another security")
        if stamp <= start or available <= start or available < stamp:
            raise ValueError("New bars must be later than initial decision and available after it")
        if previous_stamp is not None and stamp <= previous_stamp:
            raise ValueError("New bars must have strictly increasing timestamps")
        if available <= previous_available:
            raise ValueError("New bars must have strictly increasing availability")
        events.append(available)
        previous_stamp = stamp
        previous_available = available

    model.initialize(initial)
    for index, decision_at in enumerate(events):
        model.update(new_bars.slice(index, 1), decision_at=decision_at)
        forecast = model.forecast(
            decision_at=decision_at,
            upper_barrier_pct=initial.upper_barrier_pct,
            lower_barrier_pct=initial.lower_barrier_pct,
            max_holding_minutes=initial.max_holding_minutes,
        )
        # Reuse the public forecast validation rather than duplicating it.
        current = BarrierModelInput(
            security_id=initial.security_id,
            decision_at=decision_at,
            minute_bars=pa.table({}),
            upper_barrier_pct=initial.upper_barrier_pct,
            lower_barrier_pct=initial.lower_barrier_pct,
            max_holding_minutes=initial.max_holding_minutes,
        )
        validated = run_barrier_model(_ForecastAdapter(forecast), current)
        decision = decide_barrier_entry(
            validated,
            config,
            has_open_position=has_open_position,
            has_pending_entry=has_pending_entry,
        )
        yield BarrierSignalResult(forecast=validated, decision=decision)


class _ForecastAdapter:
    def __init__(self, forecast: BarrierForecast) -> None:
        self.forecast = forecast

    def predict(self, data: BarrierModelInput) -> BarrierForecast:
        return self.forecast
