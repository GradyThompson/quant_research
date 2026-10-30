"""Confidence filtering through the public rolling entry policy."""

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
        generated_at=datetime(2025, 1, 13, 15, tzinfo=UTC),
        probability_upper_first=0.65,
        probability_lower_first=0.25,
        probability_neither=0.10,
        confidence=confidence,
    )


@pytest.mark.parametrize(
    ("confidence", "expected", "reason"),
    [
        (0.85, BarrierAction.ENTER_LONG, "passes"),
        (0.80, BarrierAction.ENTER_LONG, "passes"),
        (0.79, BarrierAction.SKIP, "below threshold"),
        (None, BarrierAction.SKIP, "missing"),
    ],
)
def test_confidence_threshold(
    confidence: float | None, expected: BarrierAction, reason: str
) -> None:
    result = decide_barrier_entry(_forecast(confidence), BarrierDecisionConfig(min_confidence=0.80))
    assert result.action is expected
    assert reason in result.reason


def test_confidence_disabled_preserves_existing_policy() -> None:
    result = decide_barrier_entry(_forecast(None), BarrierDecisionConfig(min_confidence=None))
    assert result.action is BarrierAction.ENTER_LONG


def test_reject_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        BarrierDecisionConfig(min_confidence=1.1)
