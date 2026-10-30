"""Validate public barrier forecast checks using synthetic models only."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pyarrow as pa
import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.model.barrier_runner import run_barrier_model


class FixedModel:
    def __init__(self, forecast: BarrierForecast) -> None:
        self.forecast = forecast

    def predict(self, data: BarrierModelInput) -> BarrierForecast:
        return self.forecast


@pytest.fixture
def model_input() -> BarrierModelInput:
    return BarrierModelInput(
        security_id=SecurityId("SEC-000001"),
        decision_at=datetime(2025, 1, 10, 21, 1, tzinfo=UTC),
        minute_bars=pa.table({}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )


@pytest.fixture
def valid_forecast(model_input: BarrierModelInput) -> BarrierForecast:
    return BarrierForecast(
        security_id=model_input.security_id,
        generated_at=model_input.decision_at,
        probability_upper_first=0.6,
        probability_lower_first=0.3,
        probability_neither=0.1,
    )


def test_accepts_valid_forecast(
    model_input: BarrierModelInput, valid_forecast: BarrierForecast
) -> None:
    assert run_barrier_model(FixedModel(valid_forecast), model_input) == valid_forecast


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"security_id": SecurityId("SEC-000002")}, "security"),
        ({"generated_at": datetime(2025, 1, 10, 21, 1)}, "UTC-aware"),
        ({"generated_at": datetime(2025, 1, 10, 21, 2, tzinfo=UTC)}, "decision time"),
        ({"probability_upper_first": float("nan")}, "finite"),
        ({"probability_upper_first": -0.1}, "finite"),
        ({"probability_upper_first": 1.1}, "finite"),
        ({"probability_upper_first": 0.5}, "sum to one"),
    ],
)
def test_rejects_bad_forecasts(
    model_input: BarrierModelInput,
    valid_forecast: BarrierForecast,
    changes: dict[str, object],
    message: str,
) -> None:
    bad = replace(valid_forecast, **changes)
    with pytest.raises(ValueError, match=message):
        run_barrier_model(FixedModel(bad), model_input)


def test_rejects_wrong_result_type(model_input: BarrierModelInput) -> None:
    class WrongModel:
        def predict(self, data: BarrierModelInput) -> BarrierForecast:
            return None  # type: ignore[return-value]

    with pytest.raises(ValueError, match="BarrierForecast"):
        run_barrier_model(WrongModel(), model_input)


def test_rejects_non_utc_time(
    model_input: BarrierModelInput, valid_forecast: BarrierForecast
) -> None:
    forecast = replace(
        valid_forecast,
        generated_at=model_input.decision_at.astimezone(timezone(timedelta(hours=-7))),
    )
    with pytest.raises(ValueError, match="UTC-aware"):
        run_barrier_model(FixedModel(forecast), model_input)
