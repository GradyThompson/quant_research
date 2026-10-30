"""Tests for pinned real-data barrier-label wiring."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pytest

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.labels.barrier import BarrierOutcome
from quant_research.labels.pinned_barrier import label_pinned_minute_event

ENTRY = datetime(2025, 1, 10, 15, 0, tzinfo=UTC)
DATASET = DatasetId("market.ohlcv.minute.aapl.s20250110")
VERSION = DatasetVersion(uuid4())
SECURITY = SecurityId("SEC-000001")


class FakeStore:
    def __init__(self, table: pa.Table) -> None:
        self.table = table

    def read(self, dataset_id: DatasetId, version: DatasetVersion) -> pa.Table:
        assert dataset_id == DATASET
        assert version == VERSION
        return self.table


def _table(*, hit: bool = True) -> pa.Table:
    rows = []
    for i in range(3):
        time = ENTRY + timedelta(minutes=i)
        rows.append(
            {
                "security_id": str(SECURITY),
                "timestamp": time,
                "available_at": time + timedelta(seconds=60),
                "ingested_at": datetime(2026, 10, 30, tzinfo=UTC),
                "open": 100.0,
                "high": 101.0 if hit and i == 1 else 100.2,
                "low": 99.8,
                "close": 100.0,
                "volume": 100,
                "currency": "USD",
            }
        )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.MINUTE))


def _receipt(path: Path, rows: int = 3) -> None:
    path.write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "sessions": [
                    {
                        "session": "2025-01-10",
                        "dataset_id": str(DATASET),
                        "version": str(VERSION),
                        "security_id": str(SECURITY),
                        "rows": rows,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_labels_first_upper_touch(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    result = label_pinned_minute_event(
        store=FakeStore(_table()),  # type: ignore[arg-type]
        receipt_path=receipt,
        entry_at=ENTRY,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=2,
    )
    assert result.label.outcome == BarrierOutcome.UPPER_FIRST
    assert result.label.observed_at == ENTRY + timedelta(minutes=1)
    assert result.entry_price == 100.0


def test_rejects_incomplete_future_minutes(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt, rows=3)
    with pytest.raises(ValueError, match="Insufficient future"):
        label_pinned_minute_event(
            store=FakeStore(_table(hit=False)),  # type: ignore[arg-type]
            receipt_path=receipt,
            entry_at=ENTRY,
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=4,
        )


def test_rejects_missing_entry_session(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    with pytest.raises(ValueError, match="exactly one pinned session"):
        label_pinned_minute_event(
            store=FakeStore(_table()),  # type: ignore[arg-type]
            receipt_path=receipt,
            entry_at=ENTRY - timedelta(days=1),
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=2,
        )
