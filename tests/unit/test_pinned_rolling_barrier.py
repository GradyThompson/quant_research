"""Test pinned subsequent-session validation without proprietary model code."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pytest

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.model.barrier import BarrierModelInput
from quant_research.strategy.pinned_rolling_barrier import load_following_minute_session

SESSION = date(2025, 1, 13)
SID = SecurityId("SEC-000001")


def _data() -> tuple[BarrierModelInput, pa.Table]:
    calendar = xcals.get_calendar("XNYS")
    opening = calendar.session_open(pd.Timestamp(SESSION)).to_pydatetime()
    stamp = opening + timedelta(minutes=1)
    row = {
        "security_id": str(SID),
        "timestamp": stamp,
        "available_at": stamp + timedelta(minutes=1),
        "ingested_at": datetime(2026, 10, 30, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 100,
        "currency": "USD",
    }
    table = pa.Table.from_pylist([row], schema=ohlcv_schema(frequency=BarFrequency.MINUTE))
    initial = BarrierModelInput(
        SID,
        datetime(2025, 1, 10, 21, 1, tzinfo=UTC),
        pa.table({"close": [100.0]}),
        0.005,
        0.004,
        60,
    )
    return initial, table


def _load(table: pa.Table, initial: BarrierModelInput) -> pa.Table:
    store = Mock()
    store.read.return_value = table
    return load_following_minute_session(
        store=store,
        dataset_id=DatasetId("market.ohlcv.minute.aapl"),
        version=DatasetVersion(uuid4()),
        initial=initial,
        session=SESSION,
    )


def test_following_session_succeeds() -> None:
    initial, table = _data()
    assert _load(table, initial).equals(table)


def test_rejects_wrong_security() -> None:
    initial, table = _data()
    index = table.schema.get_field_index("security_id")
    altered = table.set_column(
        index,
        table.schema.field(index),
        pa.array(["SEC-000002"], type=table.schema.field(index).type),
    )
    with pytest.raises(ValueError, match="different security"):
        _load(altered, initial)


def test_rejects_unavailable_before_bar_end() -> None:
    initial, table = _data()
    index = table.schema.get_field_index("available_at")
    stamp = table.column("timestamp")[0].as_py()
    altered = table.set_column(
        index,
        table.schema.field(index),
        pa.array([stamp - timedelta(seconds=1)], type=table.schema.field(index).type),
    )
    with pytest.raises(ValueError, match="availability"):
        _load(altered, initial)


def test_rejects_initialization_overlapping_session() -> None:
    initial, table = _data()
    late_initial = BarrierModelInput(
        initial.security_id,
        datetime(2025, 1, 13, 15, 0, tzinfo=UTC),
        initial.minute_bars,
        initial.upper_barrier_pct,
        initial.lower_barrier_pct,
        initial.max_holding_minutes,
    )
    with pytest.raises(ValueError, match="after initialization"):
        _load(table, late_initial)
