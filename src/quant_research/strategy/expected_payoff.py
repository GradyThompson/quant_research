"""Strategy-neutral, three-outcome research payoff and trade selection.

All returns and costs are fractions of entry notional. This is a hypothetical
payoff calculation, NOT a simulated fill, broker order, or live executable EV.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from quant_research.model.conditional_first_passage import ThreeWayForecast


class TradeDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


@dataclass(frozen=True, slots=True)
class BarrierPayoff:
    upper_barrier_return: float
    lower_barrier_return: float
    timeout_underlying_return: float
    round_trip_cost_bps: float

    def __post_init__(self) -> None:
        if not (isfinite(self.upper_barrier_return) and 0 < self.upper_barrier_return < 1):
            raise ValueError("upper barrier must be a positive fraction")
        if not (isfinite(self.lower_barrier_return) and 0 < self.lower_barrier_return < 1):
            raise ValueError("lower barrier must be a positive fraction")
        if not isfinite(self.timeout_underlying_return):
            raise ValueError("timeout conditional return must be finite and explicitly supplied")
        if not isfinite(self.round_trip_cost_bps) or self.round_trip_cost_bps < 0:
            raise ValueError("round trip cost must be nonnegative")


@dataclass(frozen=True, slots=True)
class PayoffEvaluation:
    long_net_return: float
    short_net_return: float
    direction: TradeDirection
    selected_net_return: float


def evaluate_expected_payoff(
    forecast: ThreeWayForecast,
    payoff: BarrierPayoff,
    *,
    min_edge_bps: float = 0.0,
    allow_short: bool = False,
) -> PayoffEvaluation:
    """Rank research candidates using all three mutually exclusive outcomes.

    No-trade if the expected net return does not *exceed* the minimum edge.
    The timeout return must be independently estimated, never silently zeroed.
    """
    if not isfinite(min_edge_bps) or min_edge_bps < 0:
        raise ValueError("minimum edge must be nonnegative")
    cost = payoff.round_trip_cost_bps / 10_000.0
    underlying = (
        forecast.upper_first * payoff.upper_barrier_return
        - forecast.lower_first * payoff.lower_barrier_return
        + forecast.neither * payoff.timeout_underlying_return
    )
    long_net = underlying - cost
    short_net = -underlying - cost
    threshold = min_edge_bps / 10_000.0
    if long_net > threshold and (not allow_short or long_net >= short_net):
        direction = TradeDirection.LONG
        selected = long_net
    elif allow_short and short_net > threshold:
        direction = TradeDirection.SHORT
        selected = short_net
    else:
        direction = TradeDirection.NO_TRADE
        selected = 0.0
    return PayoffEvaluation(long_net, short_net, direction, selected)
