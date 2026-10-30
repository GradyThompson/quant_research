"""Tests for public barrier entry policy without private model logic."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast
from quant_research.strategy.barrier_decision import (
    BarrierAction,
    BarrierDecisionConfig,
    decide_barrier_entry,
)


def forecast(upper: float = 0.65, lower: float = 0.25, neither: float = 0.10) -> BarrierForecast:
    return BarrierForecast(
        security_id=SecurityId("SEC-000001"),
        generated_at=datetime(2025, 1, 10, 15, 0, tzinfo=UTC),
        probability_upper_first=upper,
        probability_lower_first=lower,
        probability_neither=neither,
    )


def test_enter_when_thresholds_met() -> None:
    decision = decide_barrier_entry(forecast(), BarrierDecisionConfig())
    assert decision.action is BarrierAction.ENTER_LONG


def test_thresholds_are_inclusive() -> None:
    decision = decide_barrier_entry(forecast(0.60, 0.35, 0.05), BarrierDecisionConfig())
    assert decision.action is BarrierAction.ENTER_LONG


def test_low_upper_probability_skips() -> None:
    decision = decide_barrier_entry(forecast(0.55, 0.25, 0.20), BarrierDecisionConfig())
    assert decision.action is BarrierAction.SKIP


def test_high_lower_probability_valid_forecast_skips() -> None:
    decision = decide_barrier_entry(forecast(0.60, 0.38, 0.02), BarrierDecisionConfig())
    assert decision.action is BarrierAction.SKIP


@pytest.mark.parametrize("flags", [(True, False), (False, True)])
def test_existing_position_or_order_skips(flags: tuple[bool, bool]) -> None:
    decision = decide_barrier_entry(
        forecast(),
        BarrierDecisionConfig(),
        has_open_position=flags[0],
        has_pending_entry=flags[1],
    )
    assert decision.action is BarrierAction.SKIP


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold(value: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        BarrierDecisionConfig(min_upper_probability=value)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((float("nan"), 0.2, 0.2), "finite"),
        ((0.7, -0.1, 0.4), "finite"),
        ((0.7, 0.3, 0.2), "sum to 1"),
    ],
)
def test_invalid_forecast(values: tuple[float, float, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decide_barrier_entry(forecast(*values), BarrierDecisionConfig())
