"""First-touch barrier outcomes from future completed minute OHLCV bars.

Research labels, not executable stop/limit fill claims. Same-bar order is
unobservable with OHLCV, so both-barrier touches are ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite

import pyarrow as pa


class BarrierOutcome(StrEnum):
    UPPER_FIRST = "upper_first"
    LOWER_FIRST = "lower_first"
    TIMEOUT = "timeout"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class BarrierLabel:
    outcome: BarrierOutcome
    observed_at: datetime
    label_available_at: datetime


def label_barrier_event(
    *,
    future_bars: pa.Table,
    entry_at: datetime,
    entry_price: float,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    max_holding_minutes: int,
) -> BarrierLabel:
    """Label the first barrier touched following an assumed entry.

    ``entry_at`` is the end of the bar immediately before the outcome window.
    Future bars must be complete, contiguous, one-minute intervals through
    the first barrier touch or timeout. No overnight/market-close gaps are
    supported in this initial implementation.

    Entry price is supplied explicitly; this does not simulate an entry fill.
    A bar touching both thresholds is AMBIGUOUS, not a win or loss.
    """
    if entry_at.tzinfo is None or entry_at.utcoffset() is None:
        raise ValueError("entry_at must be timezone-aware")
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be positive and finite")
    for name, value in (
        ("upper_barrier_pct", upper_barrier_pct),
        ("lower_barrier_pct", lower_barrier_pct),
    ):
        if not isfinite(value) or value <= 0 or value >= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if (
        isinstance(max_holding_minutes, bool)
        or not isinstance(max_holding_minutes, int)
        or max_holding_minutes < 1
    ):
        raise ValueError("max_holding_minutes must be a positive integer")

    required = {"timestamp", "available_at", "high", "low"}
    if not required.issubset(future_bars.column_names):
        raise ValueError("future_bars is missing required columns")

    upper = entry_price * (1 + upper_barrier_pct)
    lower = entry_price * (1 - lower_barrier_pct)
    previous = entry_at
    for row in future_bars.to_pylist():
        timestamp = row["timestamp"]
        available_at = row["available_at"]
        high = row["high"]
        low = row["low"]
        if not isinstance(timestamp, datetime) or not isinstance(available_at, datetime):
            raise ValueError("Future bar timestamps must be non-null datetimes")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Future bar timestamps must be timezone-aware")
        if available_at.tzinfo is None or available_at.utcoffset() is None:
            raise ValueError("Future bar availability must be timezone-aware")
        if timestamp != previous + timedelta(minutes=1):
            raise ValueError("Future minute bars must be contiguous after entry_at")
        if available_at < timestamp:
            raise ValueError("Bar availability cannot precede its end")
        if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
            raise ValueError("Future OHLC highs and lows must be numeric")
        if not isfinite(high) or not isfinite(low) or low <= 0 or high < low:
            raise ValueError("Future OHLC highs and lows are invalid")

        upper_hit = high >= upper
        lower_hit = low <= lower
        if upper_hit and lower_hit:
            outcome = BarrierOutcome.AMBIGUOUS
        elif upper_hit:
            outcome = BarrierOutcome.UPPER_FIRST
        elif lower_hit:
            outcome = BarrierOutcome.LOWER_FIRST
        else:
            outcome = None
        if outcome is not None:
            return BarrierLabel(outcome, timestamp, available_at)

        previous = timestamp
        if timestamp == entry_at + timedelta(minutes=max_holding_minutes):
            return BarrierLabel(BarrierOutcome.TIMEOUT, timestamp, available_at)

    raise ValueError("Insufficient future minute bars to determine barrier outcome")
