"""Resume pilot minute ingestion from checked market-wide flat files."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa

from quant_research.data.ingestion.minute_pilot_window import _write_json_atomic, load_plan
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.validation import OHLCVValidator

_REQUIRED = {"ticker", "window_start", "open", "high", "low", "close", "volume"}


def parse_selected_minutes(
    source: Path,
    *,
    session: date,
    tickers: set[str],
    opening: datetime,
    closing: datetime,
    publication_lag_seconds: int,
    ingested_at: datetime,
) -> dict[str, list[dict[str, object]]]:
    """Stream gzip once, keeping only the requested tickers in memory."""
    records: dict[str, list[dict[str, object]]] = {ticker: [] for ticker in tickers}
    seen: set[tuple[str, datetime]] = set()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not _REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Minute aggregate fields missing: {sorted(_REQUIRED - set(reader.fieldnames or []))}"
            )
        for raw in reader:
            ticker = raw["ticker"]
            if ticker not in tickers:
                continue
            # Massive window_start is epoch nanoseconds at the START of the minute.
            start_ns = int(raw["window_start"])
            stamp = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC) + timedelta(minutes=1)
            if not opening < stamp <= closing:
                continue
            key = (ticker, stamp)
            if key in seen:
                raise ValueError(f"Duplicate minute bar: {ticker} {stamp}")
            seen.add(key)
            from math import isfinite

            prices = [float(raw[column]) for column in ("open", "high", "low", "close")]
            volume = int(raw["volume"])
            if (
                not all(isfinite(p) and p > 0 for p in prices)
                or not prices[2]
                <= min(prices[0], prices[3])
                <= max(prices[0], prices[3])
                <= prices[1]
                or volume < 0
            ):
                raise ValueError(f"Invalid OHLCV: {ticker} {stamp}")
            records[ticker].append(
                {
                    "timestamp": stamp,
                    "available_at": stamp + timedelta(seconds=publication_lag_seconds),
                    "ingested_at": ingested_at,
                    "open": prices[0],
                    "high": prices[1],
                    "low": prices[2],
                    "close": prices[3],
                    "volume": volume,
                    "currency": "USD",
                }
            )
    for rows in records.values():
        rows.sort(key=lambda row: cast(datetime, row["timestamp"]))
    return records


def ingest_bulk_session(
    *,
    plan: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    store: ParquetDatasetStore,
    session: date,
    source: Path,
    source_sha256: str,
    publication_lag_seconds: int,
) -> int:
    """Publish outstanding ticker datasets and checkpoint each successful write."""
    entries: dict[str, dict[str, Any]] = receipt["entries"]
    todo = [
        member
        for member in plan["members"]
        if f"{member['ticker']}:{member['mic']}:{session}" not in entries
    ]
    if not todo:
        return 0
    calendar = xcals.get_calendar("XNYS")
    opening = calendar.session_open(pd.Timestamp(session)).to_pydatetime()
    closing = calendar.session_close(pd.Timestamp(session)).to_pydatetime()
    now = datetime.now(UTC)
    rows_by_ticker = parse_selected_minutes(
        source,
        session=session,
        tickers={m["ticker"] for m in todo},
        opening=opening,
        closing=closing,
        publication_lag_seconds=publication_lag_seconds,
        ingested_at=now,
    )
    # Fail closed for absent tickers: do not publish a partial or zero-row dataset.
    missing = [m["ticker"] for m in todo if not rows_by_ticker[m["ticker"]]]
    if missing:
        raise ValueError(f"No minute rows in market file for {session}: {missing}")
    result = 0
    for member in todo:
        ticker = str(member["ticker"])
        key = f"{ticker}:{member['mic']}:{session}"
        rows = [{"security_id": member["security_id"], **row} for row in rows_by_ticker[ticker]]
        table = pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.MINUTE))
        OHLCVValidator(frequency=BarFrequency.MINUTE).validate(table)
        dataset_id = session_dataset_id(ticker, session)
        metadata = store.write(dataset_id, table)
        readback = store.read(dataset_id, metadata.version)
        if readback.num_rows != len(rows) or set(readback.column("security_id").to_pylist()) != {
            member["security_id"]
        }:
            raise ValueError(f"Readback mismatch: {key}")
        entries[key] = {
            "session": session.isoformat(),
            "ticker": ticker,
            "mic": member["mic"],
            "security_id": member["security_id"],
            "dataset_id": str(dataset_id),
            "dataset_version": str(metadata.version),
            "rows": len(rows),
            "first_bar_end": rows[0]["timestamp"].isoformat(),
            "last_bar_end": rows[-1]["timestamp"].isoformat(),
            "source": "massive_minute_flat_file",
            "source_sha256": source_sha256,
        }
        _write_json_atomic(receipt_path, receipt)
        result += 1
    return result


def validate_bulk_receipt(
    plan_path: Path, receipt_path: Path, lag: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_plan(plan_path)
    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError("Plan checksum does not match receipt")
    if receipt.get("publication_lag_seconds") != lag:
        raise ValueError("Publication lag mismatch")
    if not isinstance(receipt.get("entries"), dict):
        raise ValueError("Missing receipt entries")
    return plan, receipt
