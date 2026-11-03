"""Resolve reproducible minute-backtest inputs from local dataset storage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage.base import DatasetStore


def resolve_initial_decision_at(receipt: Path) -> datetime:
    """Use one minute after the last pinned exchange-session close."""
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    sessions = raw.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Receipt contains no initialization sessions")
    last = max(date.fromisoformat(item["session"]) for item in sessions)
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(pd.Timestamp(last)):
        raise ValueError(f"Last initialization date is not a trading session: {last}")
    return calendar.session_close(pd.Timestamp(last)).to_pydatetime().astimezone(UTC) + timedelta(
        minutes=1
    )


def resolve_update_dataset(
    *,
    store: DatasetStore,
    receipt: Path,
    session: date,
    dataset_id: str | None = None,
    version: DatasetVersion | None = None,
) -> tuple[DatasetId, DatasetVersion]:
    """Find the latest stored version for this session, or a specified version.

    Explicit versions are not silently substituted when absent. Automatic
    selection is reported by the CLI, so runs can be reproduced later.
    """
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    ticker = raw.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("Receipt is missing its ticker")
    expected = f"market.ohlcv.minute.{ticker.lower()}.s{session:%Y%m%d}"
    # Include an older unsuffixed dataset ID for installations using that layout.
    candidate_ids = (
        [DatasetId(dataset_id)]
        if dataset_id
        else [
            DatasetId(expected),
            DatasetId(f"market.ohlcv.minute.{ticker.lower()}"),
        ]
    )
    for candidate in candidate_ids:
        versions = store.list_versions(candidate)
        if version is not None:
            if any(metadata.version == version for metadata in versions):
                return candidate, version
        elif versions:
            return candidate, versions[-1].version
    requested = f"version {version}" if version else "any version"
    checked = ", ".join(str(candidate) for candidate in candidate_ids)
    raise ValueError(
        f"No stored minute dataset for {ticker} on {session} ({requested}). "
        f"Checked: {checked}. Check --store-root and ingestion output; "
        "the original five-session receipt does not include this later session."
    )
