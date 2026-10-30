"""Create fixed-size, long-only research trade plans from entry decisions.

A plan is not an order or a fill. Barrier prices are determined only after an
execution simulator supplies the actual entry fill price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast
from quant_research.strategy.barrier_decision import BarrierAction, BarrierEntryDecision


@dataclass(frozen=True, slots=True)
class BarrierTradePlanConfig:
    """Research-only fixed sizing; downstream risk controls remain required."""

    shares: int = 10
    upper_barrier_pct: float = 0.005
    lower_barrier_pct: float = 0.004
    max_holding_minutes: int = 60

    def __post_init__(self) -> None:
        if type(self.shares) is not int or self.shares <= 0:
            raise ValueError("shares must be a positive integer")
        if type(self.max_holding_minutes) is not int or self.max_holding_minutes <= 0:
            raise ValueError("max_holding_minutes must be a positive integer")
        for name, value in (
            ("upper_barrier_pct", self.upper_barrier_pct),
            ("lower_barrier_pct", self.lower_barrier_pct),
        ):
            if not isfinite(value) or value <= 0 or value >= 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")


@dataclass(frozen=True, slots=True)
class BarrierTradePlan:
    """Intent to enter long at a future eligible executable price."""

    security_id: SecurityId
    signal_at: datetime
    shares: int
    upper_barrier_pct: float
    lower_barrier_pct: float
    max_holding_minutes: int

    def exit_levels(self, *, fill_price: float) -> tuple[float, float]:
        """Calculate target and stop from *actual* simulated entry price."""
        if not isfinite(fill_price) or fill_price <= 0:
            raise ValueError("fill_price must be finite and positive")
        return (
            fill_price * (1 + self.upper_barrier_pct),
            fill_price * (1 - self.lower_barrier_pct),
        )


def create_barrier_trade_plan(
    *,
    forecast: BarrierForecast,
    decision: BarrierEntryDecision,
    config: BarrierTradePlanConfig,
) -> BarrierTradePlan | None:
    """Return an entry intent only when the public policy says ENTER_LONG.

    The execution layer must enforce next-eligible-bar execution, session close,
    available cash, and no overlapping orders/positions. These are not checked
    by this pure planning function.
    """
    if decision.action is BarrierAction.SKIP:
        return None
    if decision.action is not BarrierAction.ENTER_LONG:
        raise ValueError("Unsupported entry action")
    at = forecast.generated_at
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise ValueError("signal timestamp must be UTC-aware")
    return BarrierTradePlan(
        security_id=forecast.security_id,
        signal_at=at,
        shares=config.shares,
        upper_barrier_pct=config.upper_barrier_pct,
        lower_barrier_pct=config.lower_barrier_pct,
        max_holding_minutes=config.max_holding_minutes,
    )
