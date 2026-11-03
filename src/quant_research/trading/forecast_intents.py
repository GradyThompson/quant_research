"""Publish research probability forecasts through existing trading contracts.

No order intents are generated. The same records can be consumed by replay,
shadow, paper, and live observers without exposing broker capabilities.
"""

from __future__ import annotations

from datetime import UTC, datetime

from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.trading.contracts import ForecastIntent, Instrument, Prediction


def first_touch_forecast_intents(
    *,
    instrument: Instrument,
    forecast: ThreeWayForecast,
    decision_at: datetime,
    available_at: datetime,
    horizon_minutes: int,
    model_version: str,
) -> tuple[ForecastIntent, ForecastIntent, ForecastIntent]:
    """Emit three probabilities with explicit point-in-time and version metadata."""
    for stamp in (decision_at, available_at):
        if stamp.tzinfo is None or stamp.utcoffset() != UTC.utcoffset(stamp):
            raise ValueError("decision and availability times must be UTC")
    if available_at < decision_at:
        raise ValueError("forecast cannot be available before calculation")
    if type(horizon_minutes) is not int or horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if not model_version:
        raise ValueError("model_version must be nonempty")
    values = (
        ("upper_first_probability", forecast.upper_first),
        ("lower_first_probability", forecast.lower_first),
        ("neither_probability", forecast.neither),
    )
    generated = tuple(
        ForecastIntent(
            Prediction(
                instrument=instrument,
                metric=metric,
                value=value,
                observed_at=decision_at,
                available_at=available_at,
                horizon_seconds=horizon_minutes * 60,
                model_version=model_version,
            )
        )
        for metric, value in values
    )
    return (generated[0], generated[1], generated[2])
