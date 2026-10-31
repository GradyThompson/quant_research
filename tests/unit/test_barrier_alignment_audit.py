"""Synthetic alignment checks independent of private trading models."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from quant_research.backtesting.barrier_alignment_audit import audit_barrier_alignment
from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast


def test_strict_execution_start_differs_when_forecast_on_boundary() -> None:
    at = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)
    ends = [at + timedelta(minutes=i + 1) for i in range(4)]
    bars = pa.table(
        {
            "timestamp": ends,
            "available_at": ends,
            "open": [100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    f = BarrierForecast(SecurityId("SEC-000001"), ends[0], 0.3, 0.3, 0.4)
    audit = audit_barrier_alignment(forecasts=[f], bars=bars)
    assert audit.checked == 1
    assert audit.differing_entry_times == 1
    assert audit.differing_reference_prices == 1
    assert audit.examples[0].last_known_close == 100.0
    assert audit.examples[0].calibration_open == 101.0
    assert audit.examples[0].execution_open == 102.0


def test_empty_audit_and_invalid_sample_count() -> None:
    import pytest

    bars = pa.table({"timestamp": [], "available_at": [], "open": [], "close": []})
    assert audit_barrier_alignment(forecasts=[], bars=bars).checked == 0
    with pytest.raises(ValueError, match="sample_count"):
        audit_barrier_alignment(forecasts=[], bars=bars, sample_count=-1)
