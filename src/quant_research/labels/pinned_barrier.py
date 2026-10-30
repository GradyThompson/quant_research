"""Label a historical barrier event using one immutable minute-session pin."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetStore
from quant_research.labels.barrier import BarrierLabel, label_barrier_event


@dataclass(frozen=True, slots=True)
class PinnedBarrierResult:
    security_id: SecurityId
    entry_price: float
    label: BarrierLabel
    dataset_id: DatasetId
    dataset_version: DatasetVersion


def label_pinned_minute_event(
    *,
    store: DatasetStore,
    receipt_path: Path,
    entry_at: datetime,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    max_holding_minutes: int,
) -> PinnedBarrierResult:
    """Use the close of an existing minute bar as a research entry reference.

    This is NOT an executable entry or a point-in-time model prediction.
    Subsequent bars are future data, read solely to construct a historical label.
    """
    if entry_at.tzinfo is None or entry_at.utcoffset() != timedelta(0):
        raise ValueError("entry_at must be UTC-aware")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Invalid minute receipt")
    ticker = receipt.get("ticker")
    entries = receipt.get("sessions")
    if not isinstance(ticker, str) or not isinstance(entries, list):
        raise ValueError("Invalid minute receipt")
    matching = [entry for entry in entries if entry.get("session") == entry_at.date().isoformat()]
    if len(matching) != 1:
        raise ValueError("Entry date must identify exactly one pinned session")
    entry = matching[0]
    dataset_id = DatasetId(entry["dataset_id"])
    if dataset_id != session_dataset_id(ticker, entry_at.date()):
        raise ValueError("Receipt dataset identity does not match session")
    version = DatasetVersion(UUID(entry["version"]))
    security_id = SecurityId(entry["security_id"])
    table = store.read(dataset_id, version)
    if table.schema != ohlcv_schema(frequency=BarFrequency.MINUTE):
        raise ValueError("Pinned dataset schema mismatch")
    if table.num_rows != entry["rows"] or table.num_rows == 0:
        raise ValueError("Pinned dataset row count mismatch")
    rows = table.to_pylist()
    if any(row["security_id"] != str(security_id) for row in rows):
        raise ValueError("Pinned dataset security mismatch")
    entry_rows = [row for row in rows if row["timestamp"] == entry_at]
    if len(entry_rows) != 1:
        raise ValueError("Entry timestamp must match one completed minute bar")
    reference = entry_rows[0]
    if reference["available_at"] < entry_at:
        raise ValueError("Invalid entry bar availability")
    entry_price = float(reference["close"])
    end_at = entry_at + timedelta(minutes=max_holding_minutes)
    future_rows = [row for row in rows if entry_at < row["timestamp"] <= end_at]
    future_rows.sort(key=lambda row: row["timestamp"])
    future_bars = pa.Table.from_pylist(
        future_rows,
        schema=table.schema,
    )
    label = label_barrier_event(
        future_bars=future_bars,
        entry_at=entry_at,
        entry_price=entry_price,
        upper_barrier_pct=upper_barrier_pct,
        lower_barrier_pct=lower_barrier_pct,
        max_holding_minutes=max_holding_minutes,
    )
    return PinnedBarrierResult(security_id, entry_price, label, dataset_id, version)
