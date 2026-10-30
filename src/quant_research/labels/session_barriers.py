"""Research first-touch labels across one pinned minute trading session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetStore
from quant_research.labels.barrier import BarrierLabel, label_barrier_event


@dataclass(frozen=True, slots=True)
class SessionBarrierEvent:
    entry_at: datetime
    reference_close: float
    label: BarrierLabel


@dataclass(frozen=True, slots=True)
class SessionBarrierBatch:
    security_id: SecurityId
    dataset_id: DatasetId
    dataset_version: DatasetVersion
    events: tuple[SessionBarrierEvent, ...]
    skipped_incomplete: int


def label_pinned_session(
    *,
    store: DatasetStore,
    receipt_path: Path,
    session: date,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    max_holding_minutes: int,
    stride_minutes: int = 5,
) -> SessionBarrierBatch:
    """Label entry-reference closes at fixed minute intervals within one session.

    Incomplete future windows are counted and skipped. No cross-session labels
    or executable entry-price claims are made. This function reads future bars
    only for historical labeling; they must never be passed to the predictor.
    """
    if isinstance(stride_minutes, bool) or stride_minutes < 1:
        raise ValueError("stride_minutes must be a positive integer")
    if isinstance(max_holding_minutes, bool) or max_holding_minutes < 1:
        raise ValueError("max_holding_minutes must be positive")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Invalid minute receipt")
    ticker = receipt.get("ticker")
    entries = receipt.get("sessions")
    if not isinstance(ticker, str) or not isinstance(entries, list):
        raise ValueError("Invalid minute receipt")
    matches = [
        item
        for item in entries
        if isinstance(item, dict) and item.get("session") == session.isoformat()
    ]
    if len(matches) != 1:
        raise ValueError("Session must identify exactly one pinned dataset")
    item = matches[0]
    dataset_id = DatasetId(item["dataset_id"])
    if dataset_id != session_dataset_id(ticker, session):
        raise ValueError("Receipt dataset identity mismatch")
    version = DatasetVersion(UUID(item["version"]))
    security_id = SecurityId(item["security_id"])
    table = store.read(dataset_id, version)
    if table.schema != ohlcv_schema(frequency=BarFrequency.MINUTE):
        raise ValueError("Pinned dataset schema mismatch")
    if not table.num_rows or table.num_rows != item["rows"]:
        raise ValueError("Pinned dataset row count mismatch")
    rows = table.to_pylist()
    if any(row["security_id"] != str(security_id) for row in rows):
        raise ValueError("Pinned dataset security mismatch")
    times = [row["timestamp"] for row in rows]
    if any(not isinstance(value, datetime) for value in times):
        raise ValueError("Invalid bar timestamp")
    if times != sorted(set(times)):
        raise ValueError("Minute bars must have unique ascending timestamps")
    if any(value.date() != session for value in times):
        raise ValueError("Dataset contains bars outside requested UTC date")

    events: list[SessionBarrierEvent] = []
    skipped = 0
    for index in range(0, len(rows), stride_minutes):
        reference = rows[index]
        entry_at = reference["timestamp"]
        if reference["available_at"] < entry_at:
            raise ValueError("Reference bar availability precedes bar end")
        deadline = entry_at + timedelta(minutes=max_holding_minutes)
        if deadline > times[-1]:
            skipped += 1
            continue
        future = [row for row in rows[index + 1 :] if row["timestamp"] <= deadline]
        future_table = pa.Table.from_pylist(future, schema=table.schema)
        try:
            label = label_barrier_event(
                future_bars=future_table,
                entry_at=entry_at,
                entry_price=float(reference["close"]),
                upper_barrier_pct=upper_barrier_pct,
                lower_barrier_pct=lower_barrier_pct,
                max_holding_minutes=max_holding_minutes,
            )
        except ValueError as exc:
            if "contiguous" not in str(exc) and "Insufficient future minute" not in str(exc):
                raise
            skipped += 1
            continue
        events.append(SessionBarrierEvent(entry_at, float(reference["close"]), label))

    return SessionBarrierBatch(security_id, dataset_id, version, tuple(events), skipped)
