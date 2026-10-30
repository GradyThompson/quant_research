"""Test stored snapshot selection and explicit leakage guard."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pyarrow as pa
import pytest

from quant_research.backtesting.runner import (
    DatasetPin,
    read_pinned_daily_bars,
    read_pinned_daily_data,
)
from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.schemas import BarFrequency, ohlcv_schema

T = datetime(2025, 1, 2, 21, tzinfo=UTC)


def table(*, lag: int = 0, security_id: str = "SEC-000001") -> pa.Table:
    rows = []
    for index in range(2):
        timestamp = T + timedelta(days=index)
        rows.append(
            {
                "security_id": security_id,
                "timestamp": timestamp,
                "available_at": timestamp + timedelta(minutes=lag),
                "ingested_at": timestamp + timedelta(days=1),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 100,
                "currency": "USD",
            }
        )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def pin() -> DatasetPin:
    return DatasetPin(DatasetId("market.ohlcv.daily"), DatasetVersion(uuid4()))


def test_pins_are_forwarded_to_storage_and_sorted():
    p = pin()
    store = Mock()
    store.read.return_value = table()
    bars = read_pinned_daily_bars(store, [p])
    store.read.assert_called_once_with(p.dataset_id, p.version)
    assert len(bars) == 2
    assert bars[0].timestamp < bars[1].timestamp


def test_lagged_bars_have_explicit_availability():
    store = Mock()
    store.read.return_value = table(lag=15)
    bars, available = read_pinned_daily_data(store, [pin()])
    assert available[(bars[0].security_id, bars[0].timestamp)] == bars[0].timestamp + timedelta(
        minutes=15
    )


def test_duplicate_security_timestamp_across_datasets_rejected():
    store = Mock()
    store.read.return_value = table()
    with pytest.raises(ValueError, match="duplicate security/time"):
        read_pinned_daily_bars(store, [pin(), pin()])


def test_missing_pins_rejected():
    with pytest.raises(ValueError, match="pin"):
        read_pinned_daily_bars(Mock(), [])


def test_missing_close_rejected():
    rows = table().to_pylist()
    rows[0]["close"] = None
    store = Mock()
    store.read.return_value = pa.Table.from_pylist(
        rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY)
    )
    with pytest.raises(ValueError, match="non-null OHLC prices"):
        read_pinned_daily_bars(store, [pin()])
