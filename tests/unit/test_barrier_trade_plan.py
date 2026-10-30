"""Fixed-size trade intent tests; no private model or execution assumptions."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast
from quant_research.strategy.barrier_decision import BarrierAction, BarrierEntryDecision
from quant_research.strategy.barrier_trade_plan import (
    BarrierTradePlanConfig,
    create_barrier_trade_plan,
)


def _forecast() -> BarrierForecast:
    return BarrierForecast(
        security_id=SecurityId("SEC-000001"),
        generated_at=datetime(2025, 1, 13, 15, 0, tzinfo=UTC),
        probability_upper_first=0.7,
        probability_lower_first=0.2,
        probability_neither=0.1,
        confidence=0.9,
    )


def test_accepted_entry_builds_fixed_size_plan() -> None:
    config = BarrierTradePlanConfig(shares=25, max_holding_minutes=30)
    plan = create_barrier_trade_plan(
        forecast=_forecast(),
        decision=BarrierEntryDecision(BarrierAction.ENTER_LONG, "pass"),
        config=config,
    )
    assert plan is not None
    assert plan.security_id == _forecast().security_id
    assert plan.shares == 25
    assert plan.max_holding_minutes == 30
    assert plan.exit_levels(fill_price=100.0) == pytest.approx((100.5, 99.6))


def test_skip_produces_no_plan() -> None:
    assert (
        create_barrier_trade_plan(
            forecast=_forecast(),
            decision=BarrierEntryDecision(BarrierAction.SKIP, "low confidence"),
            config=BarrierTradePlanConfig(),
        )
        is None
    )


@pytest.mark.parametrize("shares", [0, -2, 1.2, True])
def test_invalid_size(shares: object) -> None:
    with pytest.raises(ValueError, match="shares"):
        BarrierTradePlanConfig(shares=shares)  # type: ignore[arg-type]


@pytest.mark.parametrize("minutes", [0, -1, 1.5, True])
def test_invalid_horizon(minutes: object) -> None:
    with pytest.raises(ValueError, match="max_holding_minutes"):
        BarrierTradePlanConfig(max_holding_minutes=minutes)  # type: ignore[arg-type]


@pytest.mark.parametrize("barrier", [0.0, -0.1, 1.0, float("nan"), float("inf")])
def test_invalid_barrier(barrier: float) -> None:
    with pytest.raises(ValueError, match="barrier"):
        BarrierTradePlanConfig(upper_barrier_pct=barrier)
    with pytest.raises(ValueError, match="barrier"):
        BarrierTradePlanConfig(lower_barrier_pct=barrier)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_fill_prices(price: float) -> None:
    plan = create_barrier_trade_plan(
        forecast=_forecast(),
        decision=BarrierEntryDecision(BarrierAction.ENTER_LONG, "pass"),
        config=BarrierTradePlanConfig(),
    )
    assert plan is not None
    with pytest.raises(ValueError, match="fill_price"):
        plan.exit_levels(fill_price=price)
