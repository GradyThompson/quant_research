"""Tests for trailing-return feature correctness and timestamp behavior."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.exceptions import DataValidationError
from quant_research.features import TrailingReturnFeature


def make_bars(records: list[tuple[str, int, float | None]]) -> pa.Table:
    start = datetime(2025, 1, 1, 21, tzinfo=UTC)
    rows = []
    for sid, day, price in records:
        ts = start + timedelta(days=day)
        rows.append(
            {
                "security_id": sid,
                "timestamp": ts,
                "available_at": ts + timedelta(minutes=15),
                "ingested_at": ts + timedelta(minutes=16),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 100,
                "currency": "USD",
            }
        )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def test_trailing_returns_and_warmup():
    bars = make_bars([("SEC-000001", 0, 100), ("SEC-000001", 1, 110), ("SEC-000001", 2, 121)])
    result = TrailingReturnFeature(lookback=2).compute(bars)
    values = result.column("trailing_return").to_pylist()
    assert values[:2] == [None, None]
    assert values[2] == pytest.approx(0.21)


def test_interleaved_securities_are_independent_and_sorted():
    bars = make_bars(
        [
            ("SEC-000002", 1, 30),
            ("SEC-000001", 1, 110),
            ("SEC-000002", 0, 20),
            ("SEC-000001", 0, 100),
        ]
    )
    out = TrailingReturnFeature(lookback=1).compute(bars).to_pylist()
    assert [(r["security_id"], r["trailing_return"]) for r in out] == [
        ("SEC-000001", None),
        ("SEC-000001", pytest.approx(0.1)),
        ("SEC-000002", None),
        ("SEC-000002", pytest.approx(0.5)),
    ]


def test_missing_close_resets_warmup():
    bars = make_bars([("SEC-000001", 0, 10), ("SEC-000001", 1, None), ("SEC-000001", 2, 20)])
    result = TrailingReturnFeature(lookback=1).compute(bars)
    assert result.column("trailing_return").to_pylist() == [None, None, None]


def test_availability_is_not_moved_back():
    bars = make_bars([("SEC-000001", 0, 100), ("SEC-000001", 1, 90)])
    output = TrailingReturnFeature(lookback=1).compute(bars)
    assert output.column("available_at").to_pylist() == bars.column("available_at").to_pylist()


def test_duplicate_bars_rejected():
    bars = make_bars([("SEC-000001", 0, 100), ("SEC-000001", 0, 101)])
    with pytest.raises(DataValidationError):
        TrailingReturnFeature(lookback=1).compute(bars)


@pytest.mark.parametrize("lookback", [0, -1, True, 2.5])
def test_invalid_lookback(lookback):
    with pytest.raises(ValueError):
        TrailingReturnFeature(lookback=lookback)


def test_empty_bars():
    out = TrailingReturnFeature().compute(make_bars([]))
    assert out.num_rows == 0
    assert out.schema.metadata[b"feature_name"] == b"trailing_return"
