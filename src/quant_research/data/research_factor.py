"""Resumable, immutable factor-minute receipt for a restricted research window.

This component does not fetch data, choose instruments, or access holdout days.
The caller supplies only dates already permitted by the research split and
injects reference lookup, minute ingestion, and pinned-version verification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FactorIdentity:
    """Exact as-of vendor identity, obtained retrospectively for research."""

    ticker: str
    mic: str
    figi: str
    security_id: str


@dataclass(frozen=True, slots=True)
class FactorPin:
    """Version-pinned and readback-verified minute dataset."""

    dataset_id: str
    dataset_version: str
    rows: int
    first_bar_end: str
    last_bar_end: str


def requested_factor_days(entries: Sequence[dict[str, object]]) -> tuple[date, ...]:
    """Keep exactly the approved asset dates, with no implicit calendar expansion."""
    days = tuple(date.fromisoformat(str(entry["session"])) for entry in entries)
    if not days or days != tuple(sorted(set(days))):
        raise ValueError("Permitted asset entries must have unique sorted dates")
    return days


def ingest_research_factor(
    *,
    receipt_path: Path,
    asset_receipt_sha256: str,
    permitted_days: tuple[date, ...],
    ticker: str,
    mic: str,
    publication_lag_seconds: int,
    lookup_identity: Callable[[date], FactorIdentity],
    ingest_session: Callable[[date, FactorIdentity], FactorPin],
    verify_existing: Callable[[dict[str, object]], None],
    max_sessions: int = 5,
    dry_run: bool = False,
) -> dict[str, object]:
    """Persist only approved sessions, with durable checkpoints and resume support.

    No network or dataset operation takes place for dry-run. For normal runs,
    every existing pin is verified before new ingestion starts. Never change
    an existing pin to a newer version or silently extend the allowed dates.
    """
    if (
        not permitted_days
        or tuple(sorted(set(permitted_days))) != permitted_days
        or max_sessions < 1
        or publication_lag_seconds < 1
    ):
        raise ValueError("Invalid permitted dates, ingestion limit or publication lag")
    if not ticker.isascii() or not ticker.isalpha() or not ticker.isupper():
        raise ValueError("Factor ticker must be uppercase ASCII letters")
    if mic not in ("XNAS", "XNYS"):
        raise ValueError("Unsupported factor listing MIC")
    if len(asset_receipt_sha256) != 64:
        raise ValueError("Invalid asset receipt SHA-256")
    requested = [day.isoformat() for day in permitted_days]
    original: dict[str, Any] = {
        "schema_version": 1,
        "asset_receipt_sha256": asset_receipt_sha256,
        "ticker": ticker,
        "mic": mic,
        "publication_lag_seconds": publication_lag_seconds,
        "permitted_sessions": requested,
        "reference_data_point_in_time_verified": False,
        "minute_data_publication_time_verified": False,
        "research_only": True,
        "entries": {},
    }
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
        if any(receipt.get(key) != value for key, value in original.items() if key != "entries"):
            raise ValueError(
                "Factor receipt metadata differs from approved immutable research plan"
            )
        if not isinstance(receipt.get("entries"), dict):
            raise ValueError("Malformed factor receipt entries")
    else:
        receipt = original
    entries: dict[str, Any] = receipt["entries"]
    approved_keys = {f"{ticker}:{mic}:{day}" for day in requested}
    if set(entries) - approved_keys:
        raise ValueError("Factor receipt contains unapproved dates/identities")
    expected_figi: str | None = None
    expected_security_id: str | None = None
    for key, raw in sorted(entries.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed factor receipt entry: {key}")
        if (
            key != f"{raw.get('ticker')}:{raw.get('mic')}:{raw.get('session')}"
            or raw.get("ticker") != ticker
            or raw.get("mic") != mic
            or not isinstance(raw.get("figi"), str)
            or not raw["figi"]
            or not isinstance(raw.get("security_id"), str)
            or not raw["security_id"]
        ):
            raise ValueError(f"Invalid pinned factor identity: {key}")
        figi = str(raw["figi"])
        security_id = str(raw["security_id"])
        if expected_figi is not None and (figi, security_id) != (
            expected_figi,
            expected_security_id,
        ):
            raise ValueError("Factor identity changes across pinned sessions")
        expected_figi, expected_security_id = figi, security_id
        if not dry_run:
            verify_existing(raw)
    missing = [day for day in permitted_days if f"{ticker}:{mic}:{day}" not in entries]
    if dry_run:
        return {
            "complete": len(entries),
            "missing": len(missing),
            "remaining_days": [day.isoformat() for day in missing],
            "new": 0,
            "dry_run": True,
        }
    from quant_research.data.ingestion.minute_pilot_window import _write_json_atomic

    # Record an empty receipt before the first provider operation. A failed
    # retrieval can be resumed without inventing or overwriting a successful pin.
    if not receipt_path.exists():
        _write_json_atomic(receipt_path, receipt)
    completed = 0
    for day in missing[:max_sessions]:
        identity = lookup_identity(day)
        if (identity.ticker, identity.mic) != (ticker, mic) or not identity.figi:
            raise ValueError(f"Vendor factor identity mismatch for {day}")
        if expected_figi is not None and (
            identity.figi != expected_figi or identity.security_id != expected_security_id
        ):
            raise ValueError(f"Factor identity changed on {day}; refusing to merge")
        pin = ingest_session(day, identity)
        if pin.rows < 1:
            raise ValueError(f"Refusing empty factor minute dataset for {day}")
        entry: dict[str, object] = {
            "session": day.isoformat(),
            "ticker": ticker,
            "mic": mic,
            "figi": identity.figi,
            "security_id": identity.security_id,
            "dataset_id": pin.dataset_id,
            "dataset_version": pin.dataset_version,
            "rows": pin.rows,
            "first_bar_end": pin.first_bar_end,
            "last_bar_end": pin.last_bar_end,
            "identity_source": "massive.date_filtered_ticker_etf_retrieved_retrospectively",
        }
        verify_existing(entry)
        key = f"{ticker}:{mic}:{day}"
        entries[key] = entry
        _write_json_atomic(receipt_path, receipt)
        expected_figi, expected_security_id = identity.figi, identity.security_id
        completed += 1
    return {
        "complete": len(entries),
        "missing": len(permitted_days) - len(entries),
        "remaining_days": [
            day.isoformat() for day in permitted_days if f"{ticker}:{mic}:{day}" not in entries
        ],
        "new": completed,
        "dry_run": False,
    }


def receipt_sha256(path: Path) -> str:
    """Fingerprint the frozen source receipt used to select permissible dates."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
