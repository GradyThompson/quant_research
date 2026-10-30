"""Build a point-in-time five-session model input from immutable minute pins."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.ingestion.minute_window import previous_sessions, session_dataset_id
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetStore
from quant_research.model.barrier import BarrierModelInput


def _required_timestamps(column: pa.ChunkedArray) -> list[datetime]:
    """Return non-null timestamps with a concrete type for validation."""
    result: list[datetime] = []
    for value in column.to_pylist():
        if not isinstance(value, datetime):
            raise ValueError("Minute dataset contains a null or invalid timestamp")
        result.append(value)
    return result


def load_minute_model_input(
    *,
    store: DatasetStore,
    receipt_path: Path,
    decision_at: datetime,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    max_holding_minutes: int,
) -> BarrierModelInput:
    """Read the five completed sessions, admitting only bars available at decision time.

    `ingested_at` records when historical data was downloaded, not when a
    historical signal could have been made. Historical timing uses available_at.
    """
    if decision_at.tzinfo is None or decision_at.utcoffset() != timedelta(0):
        raise ValueError("decision_at must be UTC-aware")
    if upper_barrier_pct <= 0 or lower_barrier_pct <= 0:
        raise ValueError("barrier percentages must be positive")
    if max_holding_minutes <= 0:
        raise ValueError("max_holding_minutes must be positive")

    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid minute receipt")
    ticker = raw["ticker"]
    entries = raw["sessions"]
    if not isinstance(ticker, str) or not isinstance(entries, list) or len(entries) != 5:
        raise ValueError("Expected exactly five pinned sessions")

    days = [date.fromisoformat(entry["session"]) for entry in entries]
    if len(set(days)) != 5 or days != list(previous_sessions(end_session=days[-1])):
        raise ValueError("Receipt must contain five ordered consecutive exchange sessions")

    cal = xcals.get_calendar("XNYS")
    close = cal.session_close(pd.Timestamp(days[-1])).to_pydatetime()
    if decision_at < close:
        raise ValueError("All five sessions must be completed before decision_at")

    tables: list[pa.Table] = []
    expected_security_id: SecurityId | None = None
    schema = ohlcv_schema(frequency=BarFrequency.MINUTE)
    for day, entry in zip(days, entries, strict=True):
        dataset_id = DatasetId(entry["dataset_id"])
        if dataset_id != session_dataset_id(ticker, day):
            raise ValueError("Receipt dataset identity does not match session")
        version = DatasetVersion(UUID(entry["version"]))
        security_id = SecurityId(entry["security_id"])
        if expected_security_id is None:
            expected_security_id = security_id
        elif security_id != expected_security_id:
            raise ValueError("Receipt mixes security identifiers")
        table = store.read(dataset_id, version)
        if table.schema != schema or table.num_rows != entry["rows"] or table.num_rows == 0:
            raise ValueError("Pinned dataset schema or row count mismatch")
        times = _required_timestamps(table.column("timestamp"))
        available = _required_timestamps(table.column("available_at"))
        securities = table.column("security_id").to_pylist()
        if any(s != str(security_id) for s in securities):
            raise ValueError("Stored bars have an unexpected security")
        opening = cal.session_open(pd.Timestamp(day)).to_pydatetime()
        closing = cal.session_close(pd.Timestamp(day)).to_pydatetime()
        if any(not opening < stamp <= closing for stamp in times):
            raise ValueError("Bar falls outside its exchange session")
        if any(
            avail < stamp or avail > decision_at
            for stamp, avail in zip(times, available, strict=True)
        ):
            raise ValueError("Stored bar was not available at decision time")
        if any(stamp > decision_at for stamp in times):
            raise ValueError("Future bar in pinned dataset")
        tables.append(table)

    assert expected_security_id is not None
    combined = pa.concat_tables(tables)
    times_all = _required_timestamps(combined.column("timestamp"))
    if times_all != sorted(set(times_all)):
        raise ValueError("Minute timestamps must be unique and strictly increasing")
    return BarrierModelInput(
        security_id=expected_security_id,
        decision_at=decision_at.astimezone(UTC),
        minute_bars=combined,
        upper_barrier_pct=upper_barrier_pct,
        lower_barrier_pct=lower_barrier_pct,
        max_holding_minutes=max_holding_minutes,
    )
