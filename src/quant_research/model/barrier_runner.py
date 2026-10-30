"""Run a barrier model using an already-validated point-in-time input."""

from __future__ import annotations

import math
from datetime import timedelta

from quant_research.model.barrier import (
    BarrierForecast,
    BarrierModel,
    BarrierModelInput,
)


def run_barrier_model(
    model: BarrierModel,
    data: BarrierModelInput,
) -> BarrierForecast:
    """Run an injected model and validate its public forecast contract."""
    forecast = model.predict(data)
    if not isinstance(forecast, BarrierForecast):
        raise ValueError("Model must return a BarrierForecast")
    if forecast.security_id != data.security_id:
        raise ValueError("Forecast security does not match input")
    generated_at = forecast.generated_at
    if generated_at.tzinfo is None or generated_at.utcoffset() != timedelta(0):
        raise ValueError("Forecast generated_at must be UTC-aware")
    if generated_at != data.decision_at:
        raise ValueError("Forecast time does not match decision time")

    probabilities = (
        forecast.probability_upper_first,
        forecast.probability_lower_first,
        forecast.probability_neither,
    )
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probabilities):
        raise ValueError("Forecast probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Forecast probabilities must sum to one")
    return forecast
