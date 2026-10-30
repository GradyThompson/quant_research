"""Conservative daily-bar execution simulator; no broker connectivity.

Orders submitted at or after a bar's close cannot trade against that bar.
Market orders fill at the next eligible open. Limit orders may improve to the
open when marketable; otherwise they fill at the limit when touched intrabar.
Daily OHLC does not reveal intrabar event ordering or order-book liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class HistoricalOrder:
    order_id: str
    security_id: str
    side: Side
    kind: OrderType
    quantity: int
    submitted_at: datetime
    limit_price: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.order_id or not self.security_id:
            raise ValueError("order_id and security_id are required")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() != UTC.utcoffset(
            self.submitted_at
        ):
            raise ValueError("submitted_at must be UTC aware")
        if self.kind is OrderType.LIMIT:
            if (
                self.limit_price is None
                or not self.limit_price.is_finite()
                or self.limit_price <= 0
            ):
                raise ValueError("limit orders require a positive finite limit_price")
        elif self.limit_price is not None:
            raise ValueError("market orders cannot have limit_price")


@dataclass(frozen=True, slots=True)
class DailyBar:
    security_id: str
    timestamp: datetime  # completed daily bar end, UTC
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None = None

    def __post_init__(self) -> None:
        if not self.security_id:
            raise ValueError("security_id is required")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != UTC.utcoffset(
            self.timestamp
        ):
            raise ValueError("timestamp must be UTC aware")
        prices = (self.open, self.high, self.low, self.close)
        if any(not p.is_finite() or p <= 0 for p in prices):
            raise ValueError("bar prices must be positive and finite")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("invalid OHLC range")
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    order_id: str
    security_id: str
    side: Side
    quantity: int
    price: Decimal
    commission: Decimal
    bar_timestamp: datetime


@dataclass(frozen=True, slots=True)
class DailyExecutionSimulator:
    """Atomic fill-or-no-fill model with no partial fills or order queue.

    `slippage_bps` worsens the simulated execution price. For limit orders,
    a fill is declined if applying slippage would violate the limit.
    Assumes each accepted order is small enough to fill in the daily bar.
    """

    commission_per_share: Decimal = Decimal("0.005")
    slippage_bps: Decimal = Decimal("2")

    def __post_init__(self) -> None:
        for value in (self.commission_per_share, self.slippage_bps):
            if not value.is_finite() or value < 0:
                raise ValueError("cost parameters must be nonnegative and finite")

    def attempt_fill(self, order: HistoricalOrder, bar: DailyBar) -> SimulatedFill | None:
        if order.security_id != bar.security_id or bar.timestamp <= order.submitted_at:
            return None
        if bar.volume == 0:
            return None
        base: Decimal
        if order.kind is OrderType.MARKET:
            base = bar.open
        else:
            assert order.limit_price is not None
            limit = order.limit_price
            if order.side is Side.BUY:
                if bar.low > limit:
                    return None
                base = min(bar.open, limit)
            else:
                if bar.high < limit:
                    return None
                base = max(bar.open, limit)
        multiplier = Decimal(1) + (self.slippage_bps / Decimal(10000)) * (
            Decimal(1) if order.side is Side.BUY else Decimal(-1)
        )
        price = base * multiplier
        if price <= 0 or not price.is_finite():
            return None
        if order.kind is OrderType.LIMIT:
            assert order.limit_price is not None
            if order.side is Side.BUY and price > order.limit_price:
                return None
            if order.side is Side.SELL and price < order.limit_price:
                return None
        return SimulatedFill(
            order_id=order.order_id,
            security_id=order.security_id,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=self.commission_per_share * order.quantity,
            bar_timestamp=bar.timestamp,
        )
