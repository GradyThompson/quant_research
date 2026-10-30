"""Confidence gate checks using synthetic forecasts only."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast
from quant_research.strategy.barrier_decision import (
    BarrierAction,
    BarrierDecisionConfig,
    decide_barrier_entry,
)


def _forecast(confidence: float | None) -> BarrierForecast:
    return BarrierForecast(
        security_id=SecurityId("SEC-000001"),
        generated_at=datetime(2025, 1, 10, 15, 0, tzinfo=UTC),
        probability_upper_first=0.65,
        probability_lower_first=0.25,
        probability_neither=0.10,
        confidence=confidence,
    )


@pytest.mark.parametrize("confidence", [None, 0.1, 0.9])
def test_default_policy_ignores_confidence(confidence: float | None) -> None:
    decision = decide_barrier_entry(_forecast(confidence), BarrierDecisionConfig())
    assert decision.action is BarrierAction.ENTER_LONG


def test_confidence_above_threshold_enters() -> None:
    decision = decide_barrier_entry(_forecast(0.85), BarrierDecisionConfig(min_confidence=0.8))
    assert decision.action is BarrierAction.ENTER_LONG


def test_confidence_equal_to_threshold_enters() -> None:
    decision = decide_barrier_entry(_forecast(0.8), BarrierDecisionConfig(min_confidence=0.8))
    assert decision.action is BarrierAction.ENTER_LONG


def test_low_confidence_skips() -> None:
    decision = decide_barrier_entry(_forecast(0.79), BarrierDecisionConfig(min_confidence=0.8))
    assert decision.action is BarrierAction.SKIP
    assert decision.reason == "forecast confidence below threshold"


def test_missing_confidence_skips_when_required() -> None:
    decision = decide_barrier_entry(_forecast(None), BarrierDecisionConfig(min_confidence=0.8))
    assert decision.action is BarrierAction.SKIP
    assert decision.reason == "forecast confidence missing"


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_confidence_threshold(invalid: float) -> None:
    with pytest.raises(ValueError, match="Decision thresholds"):
        BarrierDecisionConfig(min_confidence=invalid)


def test_existing_position_still_skips() -> None:
    decision = decide_barrier_entry(
        _forecast(0.9),
        BarrierDecisionConfig(min_confidence=0.8),
        has_open_position=True,
    )
    assert decision.action is BarrierAction.SKIP
