"""Strictly prior-session daily OHLCV liquidity screening."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from zoneinfo import ZoneInfo

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class LiquidityStats:
    last_close: float
    mean_dollar_volume: float
    valid_sessions: int
    missing_sessions: int
    reason: str | None


def screen_daily_bars(
    bars: pa.Table,
    *,
    evaluation_date: date,
    expected_sessions: tuple[date, ...],
    min_price: float = 5.0,
    min_mean_dollar_volume: float = 10_000_000.0,
    min_valid_sessions: int = 18,
) -> LiquidityStats:
    """Exclude the evaluation session and require exactly one valid bar per prior session.

    Only a historical trading-date screen; daily bar publication timestamps
    are not evidence of point-in-time vendor availability.
    """
    if not expected_sessions or sorted(set(expected_sessions)) != list(expected_sessions):
        raise ValueError("expected_sessions must be sorted and unique")
    if expected_sessions[-1] >= evaluation_date:
        raise ValueError("Lookback includes evaluation or future session")
    if not 1 <= min_valid_sessions <= len(expected_sessions):
        raise ValueError("Invalid minimum valid sessions")
    if min_price <= 0 or min_mean_dollar_volume < 0:
        raise ValueError("Invalid screening thresholds")
    if not {"timestamp", "close", "volume"}.issubset(bars.column_names):
        raise ValueError("Daily bars require timestamp, close, volume")
    lookup = set(expected_sessions)
    data: dict[date, tuple[float, float]] = {}
    duplicate: set[date] = set()
    for row in bars.select(["timestamp", "close", "volume"]).to_pylist():
        timestamp = row["timestamp"]
        if not isinstance(timestamp, datetime):
            raise ValueError("Daily timestamp must be datetime")
        # Vendor adapter maps daily bars to US market close in UTC. The session
        # date is therefore New York local, not UTC (important around DST).
        session = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        if session not in lookup:
            continue
        if session in data:
            duplicate.add(session)
        close, volume = row["close"], row["volume"]
        if (
            not isinstance(close, (int, float))
            or not isinstance(volume, (int, float))
            or not isfinite(close)
            or not isfinite(volume)
            or close <= 0
            or volume < 0
        ):
            continue
        data[session] = (float(close), float(volume))
    for session in duplicate:
        data.pop(session, None)
    valid = len(data)
    missing = len(expected_sessions) - valid
    latest = max(data) if data else None
    price = data[latest][0] if latest is not None else 0.0
    adv = sum(close * volume for close, volume in data.values()) / valid if valid else 0.0
    reason: str | None = None
    if duplicate:
        reason = "duplicate_sessions"
    elif valid < min_valid_sessions:
        reason = "insufficient_history"
    elif latest != expected_sessions[-1]:
        reason = "missing_latest_session"
    elif price < min_price:
        reason = "below_min_price"
    elif adv < min_mean_dollar_volume:
        reason = "below_min_dollar_volume"
    return LiquidityStats(price, adv, valid, missing, reason)
