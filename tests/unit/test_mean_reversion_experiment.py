"""Integration tests using a deterministic, canonical multi-security table."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.exceptions import DataValidationError
from quant_research.research import MeanReversionExperiment


def sample_bars() -> pa.Table:
    base = datetime(2025, 1, 2, 21, tzinfo=UTC)
    rows = []
    for security_no in range(1, 7):
        for day in range(12):
            timestamp = base + timedelta(days=day)
            close = 100 + security_no * day
            rows.append(
                {
                    "security_id": f"SEC-{security_no:06d}",
                    "timestamp": timestamp,
                    "available_at": timestamp + timedelta(minutes=15),
                    "ingested_at": timestamp + timedelta(minutes=16),
                    "open": float(close),
                    "high": float(close),
                    "low": float(close),
                    "close": float(close),
                    "volume": 100,
                    "currency": "USD",
                }
            )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def test_end_to_end_experiment() -> None:
    bars = sample_bars()
    result = MeanReversionExperiment(lookback=2, horizon=2).run(bars)
    assert result.signals.num_rows == bars.num_rows
    assert result.labels.num_rows == bars.num_rows
    assert result.evaluation.summary["observations"] == bars.num_rows
    assert result.evaluation.summary["valid_pairs"] == 6 * (12 - 4)
    assert result.evaluation.summary["coverage"] == pytest.approx(8 / 12)
    assert result.evaluation.summary["evaluated_dates"] == 8


def test_research_runner_rejects_invalid_input() -> None:
    with pytest.raises(DataValidationError):
        MeanReversionExperiment().run(None)  # type: ignore[arg-type]
