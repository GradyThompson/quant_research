"""Tests for multi-event historical minute labeling."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pytest

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.labels.barrier import BarrierOutcome
from quant_research.labels.session_barriers import label_pinned_session

SESSION = date(2025, 1, 10)
START = datetime(2025, 1, 10, 15, 0, tzinfo=UTC)
DATASET = DatasetId("market.ohlcv.minute.aapl.s20250110")
VERSION = DatasetVersion(uuid4())
SECURITY = SecurityId("SEC-000001")


class FakeStore:
    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.reads = 0

    def read(self, dataset_id: DatasetId, version: DatasetVersion) -> pa.Table:
        assert dataset_id == DATASET and version == VERSION
        self.reads += 1
        return self.table


def _table(*, gap: bool = False) -> pa.Table:
    rows = []
    for index in range(6):
        minute = index + (1 if gap and index >= 2 else 0)
        timestamp = START + timedelta(minutes=minute)
        rows.append(
            {
                "security_id": str(SECURITY),
                "timestamp": timestamp,
                "available_at": timestamp + timedelta(seconds=60),
                "ingested_at": datetime(2026, 10, 30, tzinfo=UTC),
                "open": 100.0,
                "high": 101.0 if index == 1 else 100.2,
                "low": 99.8,
                "close": 100.0,
                "volume": 100,
                "currency": "USD",
            }
        )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.MINUTE))


def _receipt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "sessions": [
                    {
                        "session": SESSION.isoformat(),
                        "dataset_id": str(DATASET),
                        "version": str(VERSION),
                        "security_id": str(SECURITY),
                        "rows": 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_session_labels_multiple_events_one_read(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    store = FakeStore(_table())
    batch = label_pinned_session(
        store=store,  # type: ignore[arg-type]
        receipt_path=receipt,
        session=SESSION,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=2,
        stride_minutes=2,
    )
    assert store.reads == 1
    assert len(batch.events) == 2
    assert batch.events[0].label.outcome == BarrierOutcome.UPPER_FIRST
    assert batch.events[1].label.outcome == BarrierOutcome.TIMEOUT
    assert batch.skipped_incomplete == 1


def test_missing_minutes_marked_incomplete(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    batch = label_pinned_session(
        store=FakeStore(_table(gap=True)),  # type: ignore[arg-type]
        receipt_path=receipt,
        session=SESSION,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=2,
        stride_minutes=2,
    )
    assert batch.skipped_incomplete >= 1


def test_rejects_zero_stride(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stride_minutes"):
        label_pinned_session(
            store=FakeStore(_table()),  # type: ignore[arg-type]
            receipt_path=tmp_path / "unused.json",
            session=SESSION,
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=2,
            stride_minutes=0,
        )
