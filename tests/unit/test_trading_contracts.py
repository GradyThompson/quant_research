"""Contract-only tests: no real strategy, data service or broker required."""

from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import SecurityId
from quant_research.trading import (
    ApprovedIntent,
    Authority,
    Environment,
    ExitRule,
    Instrument,
    Market,
    Mode,
    OrderKind,
    OrderSpecification,
    PortfolioContext,
    PortfolioId,
    PortfolioPlan,
    PortfolioSnapshot,
    Prediction,
    Side,
    StrategyContext,
    StrategyDecision,
    TargetWeightIntent,
    TradePlanIntent,
)

NOW = datetime(2026, 10, 30, 15, tzinfo=UTC)
ASSET = Instrument(SecurityId("SEC-000001"), Market.US, "USD")
PID = PortfolioId("auto-us")


def test_private_strategy_and_portfolio_contracts():
    snapshot = PortfolioSnapshot(PID, NOW, "USD", 10000.0)
    context = StrategyContext(snapshot, NOW, Mode.AUTOMATED_ALPHA, Environment.BACKTEST, object())
    assert context.portfolio.portfolio_id == PID
    decision = StrategyDecision(PID, NOW, (TargetWeightIntent(ASSET, -0.1),))
    portfolio_context = PortfolioContext(snapshot, Authority.AUTOMATED, Mode.AUTOMATED_ALPHA)
    plan = PortfolioPlan(
        PID, portfolio_context.authority, (ApprovedIntent(decision.intents[0]),), ()
    )
    assert plan.approved[0].intent == decision.intents[0]


def test_manual_mode_never_has_automatic_authority():
    snapshot = PortfolioSnapshot(PID, NOW, "CAD", 10000.0)
    with pytest.raises(ValueError, match="manual alpha"):
        PortfolioContext(snapshot, Authority.AUTOMATED, Mode.MANUAL_ALPHA)


def test_limit_entry_and_contingent_exits():
    entry = OrderSpecification(Side.BUY, OrderKind.LIMIT, 100, limit_price=180.0)
    profit = ExitRule(
        "profit_target", OrderSpecification(Side.SELL, OrderKind.LIMIT, 100, limit_price=195)
    )
    stop = ExitRule(
        "protective_stop", OrderSpecification(Side.SELL, OrderKind.STOP, 100, stop_price=170)
    )
    plan = TradePlanIntent("entry-001", ASSET, entry, (profit, stop))
    assert ApprovedIntent(plan, approved_quantity=40).approved_quantity == 40
    with pytest.raises(ValueError, match="approved quantity"):
        ApprovedIntent(plan, approved_quantity=101)


def test_unresizable_entry_rejects_quantity_change():
    entry = OrderSpecification(Side.BUY, OrderKind.LIMIT, 100, limit_price=180)
    plan = TradePlanIntent("entry-001", ASSET, entry, may_reduce_quantity=False)
    with pytest.raises(ValueError, match="forbids resizing"):
        ApprovedIntent(plan, approved_quantity=50)


def test_future_portfolio_snapshots_rejected():
    snapshot = PortfolioSnapshot(PID, NOW + timedelta(minutes=1), "USD", 10000)
    with pytest.raises(ValueError, match="future"):
        StrategyContext(snapshot, NOW, Mode.AUTOMATED_ALPHA, Environment.BACKTEST, object())


def test_probabilities_are_bounded():
    with pytest.raises(ValueError, match="probability"):
        Prediction(ASSET, "upper_first_probability", 1.1, NOW, NOW, 60, "v1")


def test_trade_plan_ids_unique_within_decision():
    entry = OrderSpecification(Side.BUY, OrderKind.LIMIT, 1, limit_price=1)
    plan = TradePlanIntent("duplicate", ASSET, entry)
    with pytest.raises(ValueError, match="duplicate"):
        StrategyDecision(PID, NOW, (plan, plan))
