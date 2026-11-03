from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import SecurityId
from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.trading.contracts import (
    ForecastIntent,
    Instrument,
    Market,
    PortfolioId,
    StrategyDecision,
)
from quant_research.trading.forecast_intents import first_touch_forecast_intents

NOW = datetime(2026, 11, 1, tzinfo=UTC)
INSTRUMENT = Instrument(SecurityId("SEC-123456"), Market.US, "USD")


def test_forecast_intents_are_orderless_and_are_portfolio_compatible():
    intents = first_touch_forecast_intents(
        instrument=INSTRUMENT,
        forecast=ThreeWayForecast(0.4, 0.2, 0.4),
        decision_at=NOW,
        available_at=NOW + timedelta(seconds=2),
        horizon_minutes=10,
        model_version="m25+alpha-v1",
    )
    decision = StrategyDecision(PortfolioId("research"), NOW + timedelta(seconds=2), intents)
    assert len(decision.intents) == 3
    assert all(isinstance(intent, ForecastIntent) for intent in decision.intents)
    assert all(intent.prediction.horizon_seconds == 600 for intent in intents)
    assert {intent.prediction.metric for intent in intents} == {
        "upper_first_probability",
        "lower_first_probability",
        "neither_probability",
    }


def test_forecast_intents_reject_future_and_wrong_horizon():
    with pytest.raises(ValueError, match="before calculation"):
        first_touch_forecast_intents(
            instrument=INSTRUMENT,
            forecast=ThreeWayForecast(0.4, 0.2, 0.4),
            decision_at=NOW,
            available_at=NOW - timedelta(seconds=1),
            horizon_minutes=10,
            model_version="v",
        )
    with pytest.raises(ValueError, match="horizon_minutes"):
        first_touch_forecast_intents(
            instrument=INSTRUMENT,
            forecast=ThreeWayForecast(0.4, 0.2, 0.4),
            decision_at=NOW,
            available_at=NOW,
            horizon_minutes=0,
            model_version="v",
        )
