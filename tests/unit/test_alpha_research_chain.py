"""A complete model-to-decision chain, deliberately without broker order intents."""

from datetime import UTC, datetime, timedelta

from quant_research.core import SecurityId
from quant_research.model.conditional_first_passage import combine_first_passage_direction
from quant_research.model.directional_alpha import DirectionalExample, fit_directional_alpha
from quant_research.model.first_passage import FirstPassageProbabilities
from quant_research.strategy.expected_payoff import (
    BarrierPayoff,
    TradeDirection,
    evaluate_expected_payoff,
)
from quant_research.trading.contracts import (
    ForecastIntent,
    Instrument,
    Market,
    PortfolioId,
    StrategyDecision,
)
from quant_research.trading.forecast_intents import first_touch_forecast_intents


def test_training_to_forecast_to_advisory_decision_has_no_order_path():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = start + timedelta(days=3)
    examples = tuple(
        DirectionalExample(
            decision_at=start + timedelta(minutes=i * 2),
            label_available_at=start + timedelta(minutes=i * 2 + 1),
            feature=-2.0 + i * 0.04,
            outcome="lower" if i < 50 else "upper",
        )
        for i in range(100)
    )
    directional = fit_directional_alpha(examples, train_until=cutoff, version="m3-synthetic")
    decision_at = cutoff + timedelta(days=1)
    upper_given_touch = directional.predict_upper_given_touch(1.5, decision_at=decision_at)
    base = FirstPassageProbabilities(0.30, 0.30, 0.40)
    combined = combine_first_passage_direction(
        base, probability_upper_given_touch=upper_given_touch
    )
    payoff = evaluate_expected_payoff(
        combined,
        BarrierPayoff(0.003, 0.003, 0.0, 3),
        allow_short=False,
    )
    assert combined.neither == base.neither
    assert payoff.direction is TradeDirection.LONG
    advisory = StrategyDecision(
        portfolio_id=PortfolioId("research"),
        decided_at=decision_at,
        intents=first_touch_forecast_intents(
            instrument=Instrument(SecurityId("SEC-123456"), Market.US, "USD"),
            forecast=combined,
            decision_at=decision_at,
            available_at=decision_at,
            horizon_minutes=5,
            model_version="m25-plus-m3-synthetic",
        ),
    )
    assert len(advisory.intents) == 3
    assert all(isinstance(intent, ForecastIntent) for intent in advisory.intents)
