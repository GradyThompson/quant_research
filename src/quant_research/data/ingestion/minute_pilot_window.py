"""Resumable, pinned minute ingestion for a deterministic security pilot."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion, TimeRange
from quant_research.data.ingestion.minute import MinuteBarIngestor
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.providers.models import BarRequest, InstrumentKind, ProviderInstrument
from quant_research.data.schemas import BarFrequency
from quant_research.data.storage import ParquetDatasetStore


def load_plan(path: Path) -> dict[str, Any]:
    """Read a validated pilot plan; the plan itself is never rewritten."""
    plan: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("status") != "planned_only_no_data_downloaded":
        raise ValueError("Unexpected minute pilot plan version/status")
    dates = [date.fromisoformat(s) for s in plan["initialization_sessions"]]
    dates.append(date.fromisoformat(plan["evaluation_session"]))
    if len(dates) != len(set(dates)) or dates != sorted(dates) or len(dates) < 2:
        raise ValueError("Invalid or out-of-order minute pilot sessions")
    calendar = xcals.get_calendar("XNYS")
    if any(not calendar.is_session(pd.Timestamp(day)) for day in dates):
        raise ValueError("Pilot includes a non-trading session")
    members = plan["members"]
    if not members or len({m["security_id"] for m in members}) != len(members):
        raise ValueError("Duplicate or empty pilot members")
    if len({(m["ticker"], m["mic"]) for m in members}) != len(members):
        raise ValueError("Duplicate pilot listings")
    for member in members:
        ticker = member["ticker"]
        if (
            not ticker.isascii()
            or not ticker.isalpha()
            or not ticker.isupper()
            or member["mic"] not in ("XNAS", "XNYS")
        ):
            raise ValueError("Unsupported pilot ticker or MIC")
    if plan["planned_requests"] != len(dates) * len(members):
        raise ValueError("Planned request count mismatch")
    return plan


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish a receipt without exposing partial JSON (single process per receipt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"Stale pending receipt; inspect manually: {pending}")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    pending.replace(path)


def ingest_minute_pilot(
    *,
    plan_path: Path,
    receipt_path: Path,
    store: ParquetDatasetStore,
    ingestor: MinuteBarIngestor,
    max_requests: int = 10,
    publication_lag_seconds: int = 60,
) -> dict[str, Any]:
    """Ingest up to max_requests outstanding sessions, verifying every readback.

    Caller supplies the ingestor so tests need no network connection. Never
    replace a receipt when its existing plan fingerprint differs.
    """
    if max_requests < 1 or publication_lag_seconds < 1:
        raise ValueError("max_requests and publication_lag_seconds must be positive")
    plan = load_plan(plan_path)
    import hashlib

    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("plan_sha256") != plan_sha
            or receipt.get("publication_lag_seconds") != publication_lag_seconds
        ):
            raise ValueError("Existing receipt does not match plan or publication lag")
    else:
        receipt = {
            "schema_version": 1,
            "plan_sha256": plan_sha,
            "snapshot_version": plan["security_master_snapshot_version"],
            "publication_lag_seconds": publication_lag_seconds,
            "reference_data_point_in_time_verified": False,
            "minute_data_publication_time_verified": False,
            "entries": {},
        }
    entries: dict[str, dict[str, Any]] = receipt["entries"]
    calendar = xcals.get_calendar("XNYS")
    sessions = [date.fromisoformat(x) for x in plan["initialization_sessions"]]
    sessions.append(date.fromisoformat(plan["evaluation_session"]))
    attempted = 0
    for member in plan["members"]:
        for day in sessions:
            ticker: str = member["ticker"]
            key = f"{ticker}:{member['mic']}:{day.isoformat()}"
            dataset_id = session_dataset_id(ticker, day)
            if key in entries:
                saved_entry = entries[key]
                if (
                    saved_entry["dataset_id"] != str(dataset_id)
                    or saved_entry["security_id"] != member["security_id"]
                ):
                    raise ValueError(f"Receipt identity mismatch: {key}")
                version = DatasetVersion(UUID(saved_entry["dataset_version"]))
                saved = store.read(dataset_id, version)
                if saved.num_rows != saved_entry["rows"] or saved.num_rows == 0:
                    raise ValueError(f"Receipt dataset validation failed: {key}")
                continue
            if attempted >= max_requests:
                break
            timestamp = pd.Timestamp(day)
            opening = calendar.session_open(timestamp).to_pydatetime()
            closing = calendar.session_close(timestamp).to_pydatetime()
            request = BarRequest(
                instrument=ProviderInstrument(ticker, InstrumentKind.TICKER, member["mic"]),
                frequency=BarFrequency.MINUTE,
                period=TimeRange(opening, closing + timedelta(microseconds=1)),
            )
            result = ingestor.ingest(request, known_at=datetime.now(UTC), dataset_id=dataset_id)
            if str(result.security_id) != member["security_id"]:
                raise ValueError(
                    f"Security identity mismatch for {key}; dataset published but not receipted"
                )
            table = store.read(dataset_id, result.metadata.version)
            if table.num_rows == 0 or table.num_rows != result.metadata.row_count:
                raise ValueError(f"Minute data readback mismatch for {key}")
            expected_id = member["security_id"]
            if set(table.column("security_id").to_pylist()) != {expected_id}:
                raise ValueError(f"Minute data contains unexpected security IDs: {key}")
            entries[key] = {
                "session": day.isoformat(),
                "ticker": ticker,
                "mic": member["mic"],
                "security_id": expected_id,
                "dataset_id": str(dataset_id),
                "dataset_version": str(result.metadata.version),
                "rows": table.num_rows,
                "first_bar_end": table.column("timestamp")[0].as_py().isoformat(),
                "last_bar_end": table.column("timestamp")[-1].as_py().isoformat(),
            }
            _write_json_atomic(receipt_path, receipt)
            attempted += 1
            print(f"OK {key}: {table.num_rows} rows / {result.metadata.version}")
        if attempted >= max_requests:
            break
    receipt["completed"] = len(entries)
    receipt["total"] = plan["planned_requests"]
    receipt["remaining"] = receipt["total"] - len(entries)
    _write_json_atomic(receipt_path, receipt)
    return {"attempted": attempted, "completed": len(entries), "remaining": receipt["remaining"]}
