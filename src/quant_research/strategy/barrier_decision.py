"""Convert validated first-touch forecasts into research entry decisions.

This module does not size positions, place orders, or interpret private features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite

from quant_research.model.barrier import BarrierForecast


class BarrierAction(str, Enum):
    ENTER_LONG = "enter_long"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class BarrierDecisionConfig:
    """Fixed policy parameters; probabilities are not guaranteed win rates."""

    min_upper_probability: float = 0.6
    max_lower_probability: float = 0.35

    def __post_init__(self) -> None:
        for value in (self.min_upper_probability, self.max_lower_probability):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Decision thresholds must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class BarrierEntryDecision:
    """A research decision; downstream portfolio and risk checks are required."""

    action: BarrierAction
    reason: str


def decide_barrier_entry(
    forecast: BarrierForecast,
    config: BarrierDecisionConfig,
    *,
    has_open_position: bool = False,
    has_pending_entry: bool = False,
) -> BarrierEntryDecision:
    """Enter only for acceptable upper/lower probabilities and no existing entry.

    The policy cannot ensure a positive expected return: it does not yet account
    for slippage, spreads, timeouts, or executable entry prices.
    """
    probabilities = (
        forecast.probability_upper_first,
        forecast.probability_lower_first,
        forecast.probability_neither,
    )
    if any(not isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Forecast probabilities must be finite and within [0, 1]")
    if not isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Forecast probabilities must sum to 1")

    if has_open_position:
        return BarrierEntryDecision(BarrierAction.SKIP, "position already open")
    if has_pending_entry:
        return BarrierEntryDecision(BarrierAction.SKIP, "entry already pending")
    if forecast.probability_upper_first < config.min_upper_probability:
        return BarrierEntryDecision(BarrierAction.SKIP, "upper probability below threshold")
    if forecast.probability_lower_first > config.max_lower_probability:
        return BarrierEntryDecision(BarrierAction.SKIP, "lower probability above threshold")
    return BarrierEntryDecision(BarrierAction.ENTER_LONG, "forecast passes entry thresholds")
