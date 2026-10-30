"""Tests for the public model-to-decision boundary."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.strategy.barrier_decision import BarrierAction, BarrierDecisionConfig
from quant_research.strategy.barrier_signal import evaluate_barrier_signal


class StubModel:
    def __init__(self, upper: float = 0.65, lower: float = 0.25) -> None:
        self.upper = upper
        self.lower = lower
        self.calls = 0

    def predict(self, data: BarrierModelInput) -> BarrierForecast:
        self.calls += 1
        return BarrierForecast(
            security_id=data.security_id,
            generated_at=data.decision_at,
            probability_upper_first=self.upper,
            probability_lower_first=self.lower,
            probability_neither=1.0 - self.upper - self.lower,
        )


@pytest.fixture
def window() -> BarrierModelInput:
    return BarrierModelInput(
        security_id=SecurityId("SEC-000001"),
        decision_at=datetime(2025, 1, 10, 21, 1, tzinfo=UTC),
        minute_bars=pa.table({}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )


def test_enter_on_qualifying_forecast(window: BarrierModelInput) -> None:
    model = StubModel()
    result = evaluate_barrier_signal(model, window, BarrierDecisionConfig())
    assert model.calls == 1
    assert result.forecast.security_id == window.security_id
    assert result.decision.action is BarrierAction.ENTER_LONG


def test_skip_below_threshold(window: BarrierModelInput) -> None:
    result = evaluate_barrier_signal(StubModel(0.5, 0.3), window, BarrierDecisionConfig())
    assert result.decision.action is BarrierAction.SKIP


@pytest.mark.parametrize("flag", ["has_open_position", "has_pending_entry"])
def test_skip_when_exposure_exists(window: BarrierModelInput, flag: str) -> None:
    result = evaluate_barrier_signal(StubModel(), window, BarrierDecisionConfig(), **{flag: True})
    assert result.decision.action is BarrierAction.SKIP


def test_reject_invalid_forecast_before_policy(window: BarrierModelInput) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        evaluate_barrier_signal(StubModel(0.8, 0.7), window, BarrierDecisionConfig())
