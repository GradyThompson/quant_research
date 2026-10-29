"""Deterministic tests for forward-return labels."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.exceptions import DataValidationError
from quant_research.labels import ForwardReturnLabel

BASE = datetime(2025, 1, 2, 21, tzinfo=UTC)


def bars(securities: dict[str, list[float | None]]) -> pa.Table:
    rows = []
    for sid, closes in securities.items():
        for idx, close in enumerate(closes):
            timestamp = BASE + timedelta(days=idx)
            rows.append(
                {
                    "security_id": sid,
                    "timestamp": timestamp,
                    "available_at": timestamp + timedelta(minutes=15),
                    "ingested_at": timestamp + timedelta(minutes=16),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 100,
                    "currency": "USD",
                }
            )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def test_forward_returns_and_future_availability():
    output = (
        ForwardReturnLabel(horizon=2)
        .compute(bars({"SEC-000001": [100, 110, 121, 133.1]}))
        .to_pylist()
    )
    assert output[0]["forward_return"] == pytest.approx(0.21)
    assert output[1]["forward_return"] == pytest.approx(0.21)
    assert output[0]["label_end_time"] == BASE + timedelta(days=2)
    assert output[0]["label_available_at"] == BASE + timedelta(days=2, minutes=15)
    assert output[0]["available_at"] == BASE + timedelta(minutes=15)
    assert output[2]["forward_return"] is None
    assert output[2]["label_available_at"] is None


def test_independent_security_histories():
    output = (
        ForwardReturnLabel(horizon=1)
        .compute(bars({"SEC-000002": [200, 100], "SEC-000001": [100, 120]}))
        .to_pylist()
    )
    assert [row["security_id"] for row in output] == [
        "SEC-000001",
        "SEC-000001",
        "SEC-000002",
        "SEC-000002",
    ]
    assert output[0]["forward_return"] == pytest.approx(0.2)
    assert output[2]["forward_return"] == pytest.approx(-0.5)


def test_missing_close_invalidates_crossing_horizons():
    output = (
        ForwardReturnLabel(horizon=2)
        .compute(bars({"SEC-000001": [100, None, 120, 130, 140]}))
        .to_pylist()
    )
    assert output[0]["forward_return"] is None
    assert output[1]["forward_return"] is None
    assert output[2]["forward_return"] == pytest.approx(140 / 120 - 1)
    assert output[0]["label_end_time"] is None


def test_horizon_counts_rows_not_calendar_days():
    table = bars({"SEC-000001": [100, 105, 110]})
    rows = table.to_pylist()
    rows[1]["timestamp"] += timedelta(days=4)
    rows[1]["available_at"] += timedelta(days=4)
    rows[1]["ingested_at"] += timedelta(days=4)
    rows[2]["timestamp"] += timedelta(days=8)
    rows[2]["available_at"] += timedelta(days=8)
    rows[2]["ingested_at"] += timedelta(days=8)
    shifted = pa.Table.from_pylist(rows, schema=table.schema)
    result = ForwardReturnLabel(horizon=2).compute(shifted).to_pylist()
    assert result[0]["forward_return"] == pytest.approx(0.1)
    assert result[0]["label_end_time"] == rows[2]["timestamp"]


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        ForwardReturnLabel(horizon=horizon)


def test_empty_input():
    result = ForwardReturnLabel().compute(bars({}))
    assert result.num_rows == 0
    assert result.schema.equals(ForwardReturnLabel().schema(), check_metadata=True)


def test_invalid_input_rejected():
    table = bars({"SEC-000001": [100, 120]})
    rows = table.to_pylist()
    rows.append(rows[0].copy())
    invalid = pa.Table.from_pylist(rows, schema=table.schema)
    with pytest.raises(DataValidationError):
        ForwardReturnLabel().compute(invalid)


def test_schema_metadata():
    schema = ForwardReturnLabel(horizon=3).schema()
    assert schema.metadata is not None
    assert schema.metadata[b"horizon_observations"] == b"3"
    assert schema.metadata[b"return_semantics"] == b"close_to_future_close_not_executable"
