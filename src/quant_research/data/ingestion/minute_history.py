"""Expand a pinned minute pilot into a resumable historical training window."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.minute_pilot_window import load_plan
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.storage import ParquetDatasetStore


def expand_minute_history(
    *,
    source_plan_path: Path,
    source_receipt_path: Path,
    output_plan_path: Path,
    output_receipt_path: Path,
    store: ParquetDatasetStore,
    history_sessions: int = 60,
) -> dict[str, int]:
    """Create a longer plan, reusing only validated, version-pinned receipt entries.

    Both output paths must be unused. Existing history remains immutable; this
    function makes no provider calls. Resume subsequent ingestion with the
    standard minute-pilot ingestor and the returned new plan/receipt.
    """
    if history_sessions < 1:
        raise ValueError("history_sessions must be positive")
    if (
        output_plan_path == output_receipt_path
        or output_plan_path.exists()
        or output_receipt_path.exists()
    ):
        raise FileExistsError("Output plan/receipt path already exists")
    plan = load_plan(source_plan_path)
    old_receipt: dict[str, Any] = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if old_receipt.get("schema_version") != 1 or not isinstance(old_receipt.get("entries"), dict):
        raise ValueError("Unsupported source receipt format")
    if old_receipt.get("plan_sha256") != hashlib.sha256(source_plan_path.read_bytes()).hexdigest():
        raise ValueError("Source receipt does not match source plan")
    if old_receipt.get("snapshot_version") != plan["security_master_snapshot_version"]:
        raise ValueError("Source receipt snapshot mismatch")
    lag = old_receipt.get("publication_lag_seconds")
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("Invalid publication lag")

    calendar = xcals.get_calendar("XNYS")
    evaluation = date.fromisoformat(plan["evaluation_session"])
    idx = calendar.sessions.get_loc(pd.Timestamp(evaluation))
    if not isinstance(idx, int) or idx < history_sessions:
        raise ValueError("Insufficient calendar history")
    sessions = [s.date() for s in calendar.sessions[idx - history_sessions : idx]]
    eligible = {s.isoformat() for s in [*sessions, evaluation]}

    expanded = dict(plan)
    expanded["initialization_sessions"] = [s.isoformat() for s in sessions]
    expanded["planned_requests"] = (history_sessions + 1) * len(plan["members"])
    expanded["history_source_plan_sha256"] = hashlib.sha256(
        source_plan_path.read_bytes()
    ).hexdigest()
    expanded_bytes = (json.dumps(expanded, indent=2, sort_keys=True) + "\n").encode()
    members = {(m["ticker"], m["mic"]): m["security_id"] for m in plan["members"]}
    reused: dict[str, dict[str, Any]] = {}
    for key, entry in old_receipt["entries"].items():
        ticker, mic, session = entry["ticker"], entry["mic"], entry["session"]
        if session not in eligible:
            continue
        expected_id = members.get((ticker, mic))
        expected_key = f"{ticker}:{mic}:{session}"
        expected_dataset = session_dataset_id(ticker, date.fromisoformat(session))
        if (
            key != expected_key
            or expected_id != entry["security_id"]
            or entry["dataset_id"] != str(expected_dataset)
        ):
            raise ValueError(f"Source receipt identity mismatch: {key}")
        version = DatasetVersion(UUID(entry["dataset_version"]))
        table = store.read(expected_dataset, version)
        if table.num_rows == 0 or table.num_rows != entry["rows"]:
            raise ValueError(f"Source receipt readback mismatch: {key}")
        if set(table.column("security_id").to_pylist()) != {expected_id}:
            raise ValueError(f"Source receipt security mismatch: {key}")
        reused[key] = dict(entry)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "plan_sha256": hashlib.sha256(expanded_bytes).hexdigest(),
        "snapshot_version": plan["security_master_snapshot_version"],
        "publication_lag_seconds": lag,
        "reference_data_point_in_time_verified": False,
        "minute_data_publication_time_verified": False,
        "entries": reused,
        "completed": len(reused),
        "total": expanded["planned_requests"],
        "remaining": expanded["planned_requests"] - len(reused),
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    # Never overwrite artifacts from another run.
    output_plan_path.parent.mkdir(parents=True, exist_ok=True)
    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_plan_path.open("xb") as handle:
        handle.write(expanded_bytes)
    with output_receipt_path.open("xb") as handle:
        handle.write(receipt_bytes)
    return {
        "reused": len(reused),
        "total": expanded["planned_requests"],
        "remaining": receipt["remaining"],
    }
