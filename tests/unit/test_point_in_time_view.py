"""Prevent future bars and delayed observations leaking into model inputs."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.core import SecurityId
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.views import ArrowPointInTimeView, MarketDataRequest

T = datetime(2025, 1, 3, 21, tzinfo=UTC)
A = SecurityId("SEC-000001")
B = SecurityId("SEC-000002")


def make_bars() -> pa.Table:
    rows = []
    for sid, dt, lag in (
        (A, T - timedelta(days=1), 10),
        (A, T, 30),
        (A, T + timedelta(days=1), 10),
        (B, T - timedelta(days=1), 5),
    ):
        rows.append(
            {
                "security_id": sid.value,
                "timestamp": dt,
                "available_at": dt + timedelta(minutes=lag),
                "ingested_at": dt + timedelta(days=10),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100,
                "currency": "USD",
            }
        )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def test_filters_future_and_not_yet_available():
    view = ArrowPointInTimeView(make_bars(), decision_at=T + timedelta(minutes=15))
    result = view.bars(MarketDataRequest((A,), start=T - timedelta(days=3)))
    assert result.num_rows == 1
    assert result.column("timestamp").to_pylist() == [T - timedelta(days=1)]


def test_bars_become_visible_after_availability():
    view = ArrowPointInTimeView(make_bars(), decision_at=T + timedelta(minutes=31))
    result = view.bars(MarketDataRequest((A, B), start=T - timedelta(days=2)))
    assert result.num_rows == 3
    assert result.column("security_id").to_pylist() == [A.value, A.value, B.value]


def test_reject_incompatible_frequency_and_future_start():
    view = ArrowPointInTimeView(make_bars(), decision_at=T)
    with pytest.raises(ValueError, match="frequency or adjustment"):
        view.bars(
            MarketDataRequest((A,), start=T - timedelta(days=1), frequency=BarFrequency.MINUTE)
        )
    with pytest.raises(ValueError, match="after decision_at"):
        view.bars(MarketDataRequest((A,), start=T + timedelta(seconds=1)))


def test_empty_request_returns_canonical_schema():
    view = ArrowPointInTimeView(make_bars(), decision_at=T)
    result = view.bars(MarketDataRequest((), start=T - timedelta(days=2)))
    assert result.num_rows == 0
    assert result.schema.equals(ohlcv_schema(frequency=BarFrequency.DAILY), check_metadata=True)


def test_reject_naive_decision_time_and_duplicate_ids():
    with pytest.raises(ValueError, match="timezone-aware"):
        ArrowPointInTimeView(make_bars(), decision_at=datetime(2025, 1, 3))
    with pytest.raises(ValueError, match="duplicate"):
        MarketDataRequest((A, A), start=T)


def test_no_future_access_from_query():
    view = ArrowPointInTimeView(make_bars(), decision_at=T + timedelta(minutes=31))
    result = view.bars(MarketDataRequest((A,), start=T - timedelta(days=20)))
    assert all(row["available_at"] <= view.decision_at for row in result.to_pylist())
    assert all(row["timestamp"] <= view.decision_at for row in result.to_pylist())
