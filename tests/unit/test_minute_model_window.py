"""Offline tests for five-session, point-in-time minute window construction."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pytest

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.ingestion.minute_window import previous_sessions, session_dataset_id
from quant_research.data.schemas import BarFrequency, ohlcv_schema


class FakeStore:
    def __init__(self, tables: dict[tuple[str, str], pa.Table]) -> None:
        self.tables = tables

    def read(self, dataset_id: object, version: object) -> pa.Table:
        return self.tables[(str(dataset_id), str(version))]


def fixture(tmp_path: Path) -> tuple[FakeStore, Path, datetime]:
    calendar = xcals.get_calendar("XNYS")
    days = previous_sessions(end_session=date(2025, 1, 10))
    tables: dict[tuple[str, str], pa.Table] = {}
    entries: list[dict[str, object]] = []
    for day in days:
        opening = calendar.session_open(pd.Timestamp(day)).to_pydatetime()
        stamp = opening + timedelta(minutes=1)
        dataset = session_dataset_id("AAPL", day)
        version = DatasetVersion(uuid4())
        row = {
            "security_id": "SEC-000001",
            "timestamp": stamp,
            "available_at": stamp + timedelta(minutes=1),
            "ingested_at": datetime(2026, 10, 30, tzinfo=UTC),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100,
            "currency": "USD",
        }
        tables[(str(dataset), str(version))] = pa.Table.from_pylist(
            [row], schema=ohlcv_schema(frequency=BarFrequency.MINUTE)
        )
        entries.append(
            {
                "session": day.isoformat(),
                "dataset_id": str(dataset),
                "version": str(version),
                "security_id": "SEC-000001",
                "rows": 1,
            }
        )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"ticker": "AAPL", "sessions": entries}), encoding="utf-8")
    closing = calendar.session_close(pd.Timestamp(days[-1])).to_pydatetime()
    return FakeStore(tables), receipt, closing + timedelta(minutes=1)


def test_load_five_session_window(tmp_path: Path) -> None:
    store, receipt, decision = fixture(tmp_path)
    data = load_minute_model_input(
        store=store,  # type: ignore[arg-type]
        receipt_path=receipt,
        decision_at=decision,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )
    assert data.minute_bars.num_rows == 5
    assert str(data.security_id) == "SEC-000001"
    assert data.decision_at == decision


def test_rejects_decision_before_last_close(tmp_path: Path) -> None:
    store, receipt, decision = fixture(tmp_path)
    with pytest.raises(ValueError, match="completed"):
        load_minute_model_input(
            store=store,  # type: ignore[arg-type]
            receipt_path=receipt,
            decision_at=decision - timedelta(hours=1),
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=60,
        )


def test_rejects_unavailable_bar(tmp_path: Path) -> None:
    store, receipt, decision = fixture(tmp_path)
    key = next(reversed(store.tables))
    table = store.tables[key]
    column_idx = table.schema.get_field_index("available_at")
    updated = table.set_column(
        column_idx,
        table.schema.field(column_idx),
        pa.array([decision + timedelta(seconds=1)], type=table.schema.field(column_idx).type),
    )
    assert updated.schema == table.schema
    store.tables[key] = updated
    with pytest.raises(ValueError, match="not available"):
        load_minute_model_input(
            store=store,  # type: ignore[arg-type]
            receipt_path=receipt,
            decision_at=decision,
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=60,
        )
