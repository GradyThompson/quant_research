"""Conservative OHLC minute exits for a filled long barrier position.

Minute timestamps indicate bar END. A stop/target touched inside a bar is
filled at its level (plus sell slippage), except opening gaps, which fill at
that less-favorable opening price. Both touched within one bar resolves to
stop first. This is a research approximation, not executable tick replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

import pyarrow as pa

from quant_research.backtesting.minute_entry import MinuteEntryFill


class MinuteExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    AMBIGUOUS_STOP = "ambiguous_stop"


@dataclass(frozen=True, slots=True)
class MinuteExitConfig:
    slippage_bps: float = 2.0
    commission_per_share: float = 0.005

    def __post_init__(self) -> None:
        for value in (self.slippage_bps, self.commission_per_share):
            if not isfinite(value) or value < 0:
                raise ValueError("Execution costs must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class MinuteExitFill:
    security_id: object
    shares: int
    filled_at: datetime
    fill_price: float
    commission: float
    reason: MinuteExitReason
    gross_pnl: float
    net_pnl: float


def simulate_minute_exit(
    *,
    entry: MinuteEntryFill,
    bars: pa.Table,
    config: MinuteExitConfig | None = None,
) -> MinuteExitFill:
    """Resolve a long trade within its full, contiguous minute horizon.

    The bar beginning exactly at entry.filled_at is eligible. An incomplete
    history is rejected rather than silently treated as a timeout. The exit
    is computed only once the relevant minute OHLC is available to the
    historical simulator; this function is not intended for live orders.
    """
    if config is None:
        config = MinuteExitConfig()
    if entry.filled_at.tzinfo is None or entry.filled_at.utcoffset() != timedelta(0):
        raise ValueError("Entry time must be UTC aware")
    if entry.max_holding_minutes <= 0 or entry.shares <= 0:
        raise ValueError("Invalid entry")
    if not (0 < entry.stop_loss < entry.fill_price < entry.take_profit):
        raise ValueError("Invalid long exit levels")
    required = {"security_id", "timestamp", "open", "high", "low", "close"}
    if not required.issubset(bars.column_names):
        raise ValueError("Missing exit OHLC columns")

    deadline = entry.filled_at + timedelta(minutes=entry.max_holding_minutes)
    expected_end = entry.filled_at + timedelta(minutes=1)
    for row in bars.select(sorted(required)).to_pylist():
        end = row["timestamp"]
        if row["security_id"] != str(entry.security_id):
            raise ValueError("Exit bars contain another security")
        if not isinstance(end, datetime) or end.tzinfo is None or end.utcoffset() != timedelta(0):
            raise ValueError("Minute bar timestamps must be UTC aware")
        if end < expected_end:
            raise ValueError("Exit bars are out of order or include pre-entry data")
        if end > deadline:
            break
        if end != expected_end:
            raise ValueError("Missing contiguous future minute bars")
        prices = (row["open"], row["high"], row["low"], row["close"])
        if any(not isinstance(p, (int, float)) or not isfinite(p) or p <= 0 for p in prices):
            raise ValueError("Invalid minute OHLC")
        opening, high, low, closing = (float(p) for p in prices)
        if not (low <= min(opening, closing) <= max(opening, closing) <= high):
            raise ValueError("Inconsistent minute OHLC")

        if opening <= entry.stop_loss:
            return _fill(
                entry, config, end - timedelta(minutes=1), opening, MinuteExitReason.STOP_LOSS
            )
        if opening >= entry.take_profit:
            return _fill(
                entry,
                config,
                end - timedelta(minutes=1),
                entry.take_profit,
                MinuteExitReason.TAKE_PROFIT,
            )
        hit_stop = low <= entry.stop_loss
        hit_target = high >= entry.take_profit
        if hit_stop and hit_target:
            return _fill(entry, config, end, entry.stop_loss, MinuteExitReason.AMBIGUOUS_STOP)
        if hit_stop:
            return _fill(entry, config, end, entry.stop_loss, MinuteExitReason.STOP_LOSS)
        if hit_target:
            return _fill(entry, config, end, entry.take_profit, MinuteExitReason.TAKE_PROFIT)
        if end == deadline:
            return _fill(entry, config, end, closing, MinuteExitReason.TIMEOUT)
        expected_end += timedelta(minutes=1)
    raise ValueError("Insufficient contiguous bars to resolve the trade")


def _fill(
    entry: MinuteEntryFill,
    config: MinuteExitConfig,
    filled_at: datetime,
    reference_price: float,
    reason: MinuteExitReason,
) -> MinuteExitFill:
    price = reference_price * (1 - config.slippage_bps / 10_000)
    if price <= 0:
        raise ValueError("Sell slippage produces invalid fill price")
    commission = entry.shares * config.commission_per_share
    gross = (price - entry.fill_price) * entry.shares
    return MinuteExitFill(
        security_id=entry.security_id,
        shares=entry.shares,
        filled_at=filled_at,
        fill_price=price,
        commission=commission,
        reason=reason,
        gross_pnl=gross,
        net_pnl=gross - entry.commission - commission,
    )
