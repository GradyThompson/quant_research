"""Bounded, restartable ingestion using independently dated security-master snapshots."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from quant_research.core import DatasetId, SecurityId
from quant_research.data.ingestion.massive_pilot import daily_request
from quant_research.data.providers.models import BarRequest


class _Metadata(Protocol):
    dataset_id: DatasetId
    version: object
    row_count: int


class _Result(Protocol):
    security_id: SecurityId
    metadata: _Metadata


class _Ingestor(Protocol):
    def ingest(
        self, request: BarRequest, known_at: datetime, dataset_id: DatasetId | None = None
    ) -> _Result: ...


class _Resolver(Protocol):
    def resolve_ticker(
        self, *, ticker: str, mic: str, valid_at: datetime, known_at: datetime
    ) -> SecurityId: ...


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ingest_historical_daily_window(
    *,
    members: Sequence[Mapping[str, str]],
    snapshots: Mapping[str, str],
    manifest_path: Path,
    store_root: Path,
    make_ingestor: Callable[[str], _Ingestor],
    historical_listing: Callable[[str, str], tuple[str, str] | None],
    known_at: datetime,
    max_requests: int,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ingest at most max_requests provider calls, persisting every attempt.

    A dated resolver must be supplied by the caller for each snapshot. The
    ingestion result's security ID is checked against the evaluation universe.
    No assumption of ticker continuity or historical reference availability.
    """
    if max_requests < 1:
        raise ValueError("max_requests must be positive")
    if known_at.tzinfo is None or known_at.utcoffset() != timedelta(0):
        raise ValueError("known_at must be UTC")
    days = sorted(snapshots)
    if not days or any(date.fromisoformat(day) >= known_at.date() for day in days):
        raise ValueError("Snapshots must contain past dates")
    candidates = sorted(
        (
            {"security_id": str(m["security_id"]), "ticker": str(m["ticker"]), "mic": str(m["mic"])}
            for m in members
        ),
        key=lambda m: (m["security_id"], m["ticker"], m["mic"]),
    )
    if len({m["security_id"] for m in candidates}) != len(candidates):
        raise ValueError("Duplicate security IDs in universe")
    keys = {
        "format": 1,
        "store_root": str(store_root.resolve()),
        "snapshots": dict(sorted(snapshots.items())),
        "members": candidates,
    }
    if manifest_path.exists():
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(payload.get(k) != v for k, v in keys.items()) or not isinstance(
            payload.get("results"), dict
        ):
            raise ValueError("Manifest input mismatch; use a new manifest path")
    else:
        payload = {**keys, "results": {}, "historical_reference_availability_verified": False}
    results: dict[str, Any] = payload["results"]
    attempted = 0
    for day_text in days:
        day = date.fromisoformat(day_text)
        ingestor: _Ingestor | None = None
        for member in candidates:
            sid, ticker, mic = member["security_id"], member["ticker"], member["mic"]
            key = f"{day_text}|{sid}"
            if key in results and results[key].get("status") in {
                "ok",
                "not_listed",
                "unsupported_mic",
                "no_bars",
            }:
                continue
            listing = historical_listing(snapshots[day_text], sid)
            if listing is None:
                results[key] = {"status": "not_listed", "ticker": ticker, "mic": mic}
                _write_atomic(manifest_path, payload)
                continue
            ticker, mic = listing
            if mic not in {"XNAS", "XNYS"}:
                results[key] = {"status": "unsupported_mic", "ticker": ticker, "mic": mic}
                _write_atomic(manifest_path, payload)
                continue
            if attempted >= max_requests:
                _write_atomic(manifest_path, payload)
                return payload
            if ingestor is None:
                ingestor = make_ingestor(snapshots[day_text])
            attempted += 1
            dataset_id = DatasetId(f"market.ohlcv.daily.{ticker.lower()}.s{day:%Y%m%d}")
            try:
                response = ingestor.ingest(
                    daily_request(ticker, mic, day, day), known_at, dataset_id
                )
                if str(response.security_id) != sid:
                    raise ValueError("Resolved security differs from evaluation universe")
                results[key] = {
                    "status": "ok",
                    "ticker": ticker,
                    "mic": mic,
                    "security_id": sid,
                    "dataset_id": str(response.metadata.dataset_id),
                    "version": str(response.metadata.version),
                    "rows": response.metadata.row_count,
                    "snapshot_version": snapshots[day_text],
                }
                message = f"OK {day_text} {ticker}:{mic} -> {response.metadata.version}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                # Missing bars are normal for inactive securities; unmatched
                # listings are surfaced for later inspection, not retried.
                if "No bars returned" in str(exc):
                    status = "no_bars"
                elif "No listing" in str(exc) or "not found" in str(exc).lower():
                    status = "not_listed"
                else:
                    status = "error"
                results[key] = {"status": status, "ticker": ticker, "mic": mic, "error": error}
                message = f"{status.upper()} {day_text} {ticker}:{mic}: {error}"
            _write_atomic(manifest_path, payload)
            if on_progress is not None:
                on_progress(message)
    _write_atomic(manifest_path, payload)
    return payload
