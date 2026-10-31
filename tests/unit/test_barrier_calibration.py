"""Regression tests for calibration reference-time conventions."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from quant_research.backtesting.barrier_calibration import (
    ReferenceConvention,
    evaluate_barrier_calibration,
)
from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast


def test_reference_convention_changes_realized_outcome() -> None:
    base = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)
    stamps = [base + timedelta(minutes=i + 1) for i in range(4)]
    bars = pa.table(
        {
            "timestamp": stamps,
            "available_at": [t + timedelta(seconds=1) for t in stamps],
            "open": [100.0] * 4,
            "high": [101.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
        }
    )
    forecast = BarrierForecast(SecurityId("SEC-000001"), base, 0.7, 0.2, 0.1)
    common = dict(
        forecasts=[forecast],
        bars=bars,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.005,
        max_holding_minutes=2,
    )
    legacy = evaluate_barrier_calibration(**common, reference_convention=ReferenceConvention.LEGACY)
    executable = evaluate_barrier_calibration(**common)
    assert legacy.observed == {"upper_first": 1}
    assert executable.observed == {"timeout": 1}
    assert executable.scored == legacy.scored == 1


def test_missing_horizon_excluded() -> None:
    base = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)
    bars = pa.table(
        {
            "timestamp": [base + timedelta(minutes=1)],
            "available_at": [base + timedelta(minutes=1)],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
        }
    )
    forecast = BarrierForecast(SecurityId("SEC-000001"), base, 0.3, 0.3, 0.4)
    result = evaluate_barrier_calibration(
        forecasts=[forecast],
        bars=bars,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.005,
        max_holding_minutes=2,
    )
    assert result.incomplete == 1
    assert result.scored == 0
    assert result.brier_score is None
