"""Point-in-time minute log-return volatility, without model fitting.

Only adjacent one-minute bars form valid returns. Overnight jumps and
missing-minute gaps are excluded rather than treated as one-minute moves.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite, log, sqrt

import pyarrow as pa


def estimate_minute_volatility(
    bars: pa.Table,
    *,
    decision_at: datetime,
    lookback_returns: int = 390,
    min_returns: int = 60,
) -> float:
    """Root mean squared historical one-minute log return.

    Input must contain complete historical minute bars in strictly
    increasing timestamp order. All bars must already have been published
    at `decision_at`. The zero-mean diffusion assumption motivates using
    squared log returns without estimating a noisy intraday drift.
    """
    if decision_at.tzinfo is None or decision_at.utcoffset() != timedelta(0):
        raise ValueError("decision_at must be UTC-aware")
    if (
        isinstance(lookback_returns, bool)
        or not isinstance(lookback_returns, int)
        or lookback_returns < 1
    ):
        raise ValueError("lookback_returns must be a positive integer")
    if (
        isinstance(min_returns, bool)
        or not isinstance(min_returns, int)
        or not 1 <= min_returns <= lookback_returns
    ):
        raise ValueError("min_returns must be between 1 and lookback_returns")
    if not {"timestamp", "available_at", "close"}.issubset(bars.column_names):
        raise ValueError("Minute bars require timestamp, available_at, close")

    timestamps = bars.column("timestamp").to_pylist()
    available = bars.column("available_at").to_pylist()
    closes = bars.column("close").to_pylist()
    previous_at: datetime | None = None
    previous_close: float | None = None
    log_returns: list[float] = []
    for stamp, known_at, close in zip(timestamps, available, closes, strict=True):
        if (
            not isinstance(stamp, datetime)
            or stamp.tzinfo is None
            or stamp.utcoffset() != timedelta(0)
        ):
            raise ValueError("Minute timestamp must be UTC-aware")
        if (
            not isinstance(known_at, datetime)
            or known_at.tzinfo is None
            or known_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("Minute availability must be UTC-aware")
        if (
            (previous_at is not None and stamp <= previous_at)
            or known_at < stamp
            or stamp > decision_at
            or known_at > decision_at
        ):
            raise ValueError("Bars must be ordered, complete and available by decision_at")
        if not isinstance(close, (int, float)) or not isfinite(close) or close <= 0.0:
            raise ValueError("Minute close must be positive and finite")
        if previous_at is not None and stamp - previous_at == timedelta(minutes=1):
            assert previous_close is not None
            log_returns.append(log(close) - log(previous_close))
        previous_at = stamp
        previous_close = float(close)

    recent = log_returns[-lookback_returns:]
    if len(recent) < min_returns:
        raise ValueError(f"Insufficient contiguous minute returns: {len(recent)} < {min_returns}")
    return sqrt(sum(value * value for value in recent) / len(recent))
