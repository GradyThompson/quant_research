"""Conservative risk gate behavior; synthetic intents only."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.core import SecurityId
from quant_research.trading.contracts import (
    ApprovedIntent,
    Authority,
    CancelIntent,
    Instrument,
    Market,
    Mode,
    OrderKind,
    OrderSpecification,
    PortfolioId,
    PortfolioPlan,
    Side,
    TargetWeightIntent,
    TradePlanIntent,
)
from quant_research.trading.risk import PreTradeRiskGate, RiskContext, RiskLimits

NOW = datetime(2026, 10, 30, 16, tzinfo=UTC)
PID = PortfolioId("sandbox")


def instrument(number: int, market: Market = Market.US) -> Instrument:
    return Instrument(
        SecurityId(f"SEC-{number:06d}"), market, "USD" if market is Market.US else "CAD"
    )


def plan(mode: Mode, *intents, authority: Authority = Authority.AUTOMATED) -> PortfolioPlan:
    return PortfolioPlan(PID, authority, tuple(ApprovedIntent(intent) for intent in intents), ())


def ctx(mode: Mode, age: timedelta = timedelta(seconds=2), halted: bool = False) -> RiskContext:
    return RiskContext(mode, NOW, NOW - age, halted)


def trade(quantity: int = 10, price: float = 20.0) -> TradePlanIntent:
    return TradePlanIntent(
        "entry-1",
        instrument(1),
        OrderSpecification(Side.BUY, OrderKind.LIMIT, quantity, limit_price=price),
    )


def test_approved_trade():
    result = PreTradeRiskGate().check(plan(Mode.SHORT_TERM, trade()), ctx(Mode.SHORT_TERM))
    assert len(result.approved) == 1
    assert not result.rejected


def test_reject_large_order():
    result = PreTradeRiskGate().check(plan(Mode.SHORT_TERM, trade(300)), ctx(Mode.SHORT_TERM))
    assert not result.approved
    assert "notional" in result.rejected[0].reason


def test_stale_snapshot_blocks_entry_but_allows_cancel():
    result = PreTradeRiskGate().check(
        plan(Mode.SHORT_TERM, trade(), CancelIntent("previous")),
        ctx(Mode.SHORT_TERM, timedelta(minutes=6)),
    )
    assert len(result.rejected) == 1
    assert len(result.approved) == 1
    assert isinstance(result.approved[0].intent, CancelIntent)


def test_halt_blocks_entries_not_cancels():
    result = PreTradeRiskGate().check(
        plan(Mode.SHORT_TERM, trade(), CancelIntent("previous")), ctx(Mode.SHORT_TERM, halted=True)
    )
    assert len(result.rejected) == 1
    assert len(result.approved) == 1


def test_manual_recommendations_never_executable():
    result = PreTradeRiskGate().check(
        plan(
            Mode.MANUAL_ALPHA,
            TargetWeightIntent(instrument(1, Market.CA), 0.1),
            authority=Authority.MANUAL,
        ),
        ctx(Mode.MANUAL_ALPHA),
    )
    assert not result.approved
    assert "advisory" in result.rejected[0].reason


def test_alpha_gross_and_single_name_limits():
    gate = PreTradeRiskGate(RiskLimits(max_gross_target_weight=Decimal("0.3")))
    result = gate.check(
        plan(
            Mode.AUTOMATED_ALPHA,
            TargetWeightIntent(instrument(1), 0.2),
            TargetWeightIntent(instrument(2), -0.2),
        ),
        ctx(Mode.AUTOMATED_ALPHA),
    )
    assert len(result.rejected) == 2
    result = PreTradeRiskGate().check(
        plan(Mode.AUTOMATED_ALPHA, TargetWeightIntent(instrument(1), 0.5)),
        ctx(Mode.AUTOMATED_ALPHA),
    )
    assert len(result.rejected) == 1


def test_alpha_small_target_approved_as_allocation_only():
    result = PreTradeRiskGate().check(
        plan(Mode.AUTOMATED_ALPHA, TargetWeightIntent(instrument(1), -0.1)),
        ctx(Mode.AUTOMATED_ALPHA),
    )
    assert len(result.approved) == 1


def test_reject_unpriced_market_order():
    item = TradePlanIntent(
        "market", instrument(1), OrderSpecification(Side.BUY, OrderKind.MARKET, 10)
    )
    result = PreTradeRiskGate().check(plan(Mode.SHORT_TERM, item), ctx(Mode.SHORT_TERM))
    assert len(result.rejected) == 1


def test_invalid_limits_and_future_timestamp():
    with pytest.raises(ValueError):
        RiskLimits(max_order_notional_usd=Decimal("NaN"))
    with pytest.raises(ValueError):
        RiskContext(Mode.SHORT_TERM, NOW, NOW + timedelta(seconds=1))
