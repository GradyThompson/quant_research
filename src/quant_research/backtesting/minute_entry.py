"""Conservative next-minute market entry simulation for barrier trade plans.

A minute bar timestamp is the bar END. The simulated order fills at the
bar OPEN, so that open must occur strictly after the signal became available.
This is an atomic fill approximation, not an order-book or broker simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

import pyarrow as pa

from quant_research.core import SecurityId
from quant_research.strategy.barrier_trade_plan import BarrierTradePlan


@dataclass(frozen=True, slots=True)
class MinuteEntryConfig:
    slippage_bps: float = 2.0
    commission_per_share: float = 0.005

    def __post_init__(self) -> None:
        for value in (self.slippage_bps, self.commission_per_share):
            if not isfinite(value) or value < 0:
                raise ValueError("Execution costs must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class MinuteEntryFill:
    security_id: SecurityId
    shares: int
    signal_at: datetime
    filled_at: datetime
    fill_price: float
    commission: float
    take_profit: float
    stop_loss: float
    max_holding_minutes: int


def simulate_next_minute_entry(
    *,
    plan: BarrierTradePlan,
    bars: pa.Table,
    config: MinuteEntryConfig | None = None,
    session_close: datetime,
) -> MinuteEntryFill | None:
    """Fill at first eligible minute OPEN, otherwise return None.

    `bars` must be one security's chronological regular-session bars, with
    `timestamp` as bar END. This function never uses the current bar's open
    when its start is at or before `signal_at` (prevents lookahead).

    For now, a plan must have enough session time for its full horizon;
    otherwise no entry is permitted. Cash and exposure checks are upstream.
    """
    if config is None:
        config = MinuteEntryConfig()

    if plan.signal_at.tzinfo is None or plan.signal_at.utcoffset() != timedelta(0):
        raise ValueError("Signal must be UTC aware")
    if session_close.tzinfo is None or session_close.utcoffset() != timedelta(0):
        raise ValueError("Session close must be UTC aware")
    if plan.shares <= 0 or plan.max_holding_minutes <= 0:
        raise ValueError("Invalid trade plan")
    needed = {"security_id", "timestamp", "open"}
    if not needed.issubset(bars.column_names):
        raise ValueError("Missing minute entry columns")
    previous_end: datetime | None = None
    chosen: tuple[datetime, float] | None = None
    for row in bars.select(["security_id", "timestamp", "open"]).to_pylist():
        if row["security_id"] != str(plan.security_id):
            raise ValueError("Bar security does not match trade plan")
        end = row["timestamp"]
        price = row["open"]
        if not isinstance(end, datetime) or end.tzinfo is None or end.utcoffset() != timedelta(0):
            raise ValueError("Bar timestamps must be UTC aware")
        if previous_end is not None and end <= previous_end:
            raise ValueError("Minute bars must be strictly ordered")
        if not isinstance(price, (int, float)) or not isfinite(price) or price <= 0:
            raise ValueError("Invalid bar open")
        start = end - timedelta(minutes=1)
        if end > session_close or start >= session_close:
            raise ValueError("Bars must be within the requested session close")
        if chosen is None and start > plan.signal_at:
            chosen = start, float(price)
        previous_end = end
    if chosen is None:
        return None
    filled_at, raw_price = chosen
    if filled_at + timedelta(minutes=plan.max_holding_minutes) > session_close:
        return None
    fill_price = raw_price * (1 + config.slippage_bps / 10_000)
    target, stop = plan.exit_levels(fill_price=fill_price)
    return MinuteEntryFill(
        security_id=plan.security_id,
        shares=plan.shares,
        signal_at=plan.signal_at,
        filled_at=filled_at,
        fill_price=fill_price,
        commission=plan.shares * config.commission_per_share,
        take_profit=target,
        stop_loss=stop,
        max_holding_minutes=plan.max_holding_minutes,
    )
