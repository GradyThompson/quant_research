"""Optional confidence metadata does not change barrier outcome probabilities."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast


def _forecast(*, confidence: float | None = None) -> BarrierForecast:
    return BarrierForecast(
        security_id=SecurityId("SEC-000001"),
        generated_at=datetime(2025, 1, 10, 21, 1, tzinfo=UTC),
        probability_upper_first=0.6,
        probability_lower_first=0.3,
        probability_neither=0.1,
        confidence=confidence,
    )


def test_legacy_forecasts_default_to_unknown_confidence() -> None:
    value = _forecast()
    assert value.confidence is None
    assert value.probability_upper_first == 0.6


def test_confidence_can_be_reported() -> None:
    assert _forecast(confidence=0.85).confidence == 0.85


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_confidence_accepts_boundaries(value: float) -> None:
    assert _forecast(confidence=value).confidence == value


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _forecast(confidence=value)
