"""Three investment-mode planning contracts and safe reject behavior."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_research.core import SecurityId
from quant_research.portfolio.mode_constructors import (
    AutomatedAlphaConstructor,
    ManualAlphaConstructor,
    ShortTermTradeConstructor,
)
from quant_research.trading.capital import (
    CapitalLimit,
    CashBalance,
    Currency,
    PortfolioCapitalSnapshot,
)
from quant_research.trading.contracts import (
    Authority,
    ForecastIntent,
    Instrument,
    Market,
    Mode,
    OrderKind,
    OrderSpecification,
    PortfolioContext,
    PortfolioId,
    PortfolioSnapshot,
    Prediction,
    Side,
    StrategyDecision,
    TargetWeightIntent,
    TradePlanIntent,
)

NOW = datetime(2026, 10, 30, 16, tzinfo=UTC)
PID = PortfolioId("sandbox")


def instrument(number: int, market: Market = Market.US) -> Instrument:
    return Instrument(
        SecurityId(f"SEC-{number:06d}"), market, "USD" if market is Market.US else "CAD"
    )


def context(
    mode: Mode, authority: Authority = Authority.AUTOMATED, gross: float = 1.0
) -> PortfolioContext:
    return PortfolioContext(PortfolioSnapshot(PID, NOW, "USD", 10000.0), authority, mode, gross)


def decision(*intents):
    return StrategyDecision(PID, NOW, tuple(intents))


def capital(amount: str) -> PortfolioCapitalSnapshot:
    return PortfolioCapitalSnapshot(
        PID,
        Mode.SHORT_TERM,
        NOW,
        Currency.USD,
        (CashBalance(Currency.USD, Decimal("10000")),),
        limits=(CapitalLimit(Currency.USD, Decimal(amount)),),
    )


def trade(
    number: int, quantity: int = 10, price: float = 100.0, *, resize: bool = True
) -> TradePlanIntent:
    return TradePlanIntent(
        f"id-{number}",
        instrument(number),
        OrderSpecification(Side.BUY, OrderKind.LIMIT, quantity, limit_price=price),
        may_reduce_quantity=resize,
    )


def test_automated_alpha_approves_within_limit():
    result = AutomatedAlphaConstructor().construct(
        context(Mode.AUTOMATED_ALPHA, gross=1.5),
        decision(TargetWeightIntent(instrument(1), 0.8), TargetWeightIntent(instrument(2), -0.5)),
    )
    assert len(result.approved) == 2
    assert not result.rejected


def test_alpha_rejects_full_target_set_if_gross_exceeded():
    result = AutomatedAlphaConstructor().construct(
        context(Mode.AUTOMATED_ALPHA),
        decision(TargetWeightIntent(instrument(1), 0.8), TargetWeightIntent(instrument(2), -0.5)),
    )
    assert len(result.rejected) == 2
    assert not result.approved


def test_automated_rejects_unsupported_forecast_intent():
    forecast = ForecastIntent(
        Prediction(instrument(1), "relative_score", 0.5, NOW, NOW, 3600, "v1")
    )
    result = AutomatedAlphaConstructor().construct(
        context(Mode.AUTOMATED_ALPHA), decision(forecast)
    )
    assert len(result.rejected) == 1


def test_manual_accepts_canadian_recommendation_and_never_auto_executes():
    intent = TargetWeightIntent(instrument(1, Market.CA), 0.2)
    result = ManualAlphaConstructor().construct(
        context(Mode.MANUAL_ALPHA, Authority.MANUAL), decision(intent)
    )
    assert result.authority is Authority.MANUAL
    assert result.approved[0].intent == intent
    with pytest.raises(ValueError):
        context(Mode.MANUAL_ALPHA, Authority.AUTOMATED)


def test_short_term_sizes_conservatively_and_reserves_across_requests():
    result = ShortTermTradeConstructor().construct(
        context(Mode.SHORT_TERM), decision(trade(1, 10, 100), trade(2, 10, 100)), capital("1500")
    )
    assert [item.approved_quantity for item in result.approved] == [10, 5]
    assert not result.rejected


def test_short_term_respects_no_resize():
    result = ShortTermTradeConstructor().construct(
        context(Mode.SHORT_TERM), decision(trade(1, 10, 100, resize=False)), capital("500")
    )
    assert len(result.rejected) == 1


def test_short_term_rejects_market_orders():
    intent = TradePlanIntent(
        "market-1", instrument(1), OrderSpecification(Side.BUY, OrderKind.MARKET, 1)
    )
    result = ShortTermTradeConstructor().construct(
        context(Mode.SHORT_TERM), decision(intent), capital("1000")
    )
    assert len(result.rejected) == 1


def test_short_term_requires_capital_snapshot():
    with pytest.raises(ValueError, match="requires capital"):
        ShortTermTradeConstructor().construct(context(Mode.SHORT_TERM), decision(trade(1)))


def test_portfolio_mismatch_rejected():
    wrong = StrategyDecision(PortfolioId("different"), NOW, ())
    with pytest.raises(ValueError, match="mismatch"):
        AutomatedAlphaConstructor().construct(context(Mode.AUTOMATED_ALPHA), wrong)


def test_automated_rejects_canadian_equity():
    with pytest.raises(ValueError, match="US-equity"):
        AutomatedAlphaConstructor().construct(
            context(Mode.AUTOMATED_ALPHA),
            decision(TargetWeightIntent(instrument(1, Market.CA), 0.2)),
        )
