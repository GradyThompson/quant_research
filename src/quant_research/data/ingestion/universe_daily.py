"""Bounded, restartable single-session daily ingestion for a saved universe."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

from quant_research.core import DatasetId
from quant_research.data.ingestion.massive_pilot import daily_request
from quant_research.data.providers.models import BarRequest


class _Metadata(Protocol):
    dataset_id: DatasetId
    version: object
    row_count: int


class _Result(Protocol):
    metadata: _Metadata


class _Ingestor(Protocol):
    def ingest(
        self, request: BarRequest, known_at: datetime, dataset_id: DatasetId | None = None
    ) -> _Result: ...


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ingest_universe_daily_session(
    *,
    ingestor: _Ingestor,
    members: Sequence[Mapping[str, str]],
    session: date,
    snapshot_version: str,
    manifest: Path,
    max_requests: int,
    known_at: datetime,
    eligible_mics: frozenset[str] = frozenset({"XNAS", "XNYS"}),
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Ingest up to max_requests unique tickers; persist status after every request.

    Per-symbol provider requests intentionally bound work. A success can be
    resumed from the manifest, but existing store versions are not overwritten.
    The manifest is tied to one input inventory and trading session.
    """
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if known_at.tzinfo is None or known_at.utcoffset() != timedelta(0):
        raise ValueError("known_at must be timezone-aware UTC")
    if session >= known_at.date():
        raise ValueError("Historical session must precede known_at")
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("session") != session.isoformat()
            or payload.get("snapshot_version") != snapshot_version
        ):
            raise ValueError("Manifest does not match requested session and snapshot")
    else:
        payload = {
            "schema_version": 1,
            "session": session.isoformat(),
            "snapshot_version": snapshot_version,
            "retrieval_is_historical_point_in_time": False,
            "results": {},
        }
    results = payload["results"]
    if not isinstance(results, dict):
        raise ValueError("Malformed manifest results")
    attempted = 0
    seen: set[str] = set()
    for member in members:
        ticker, mic = member["ticker"], member["mic"]
        # Dedupe tickers because the existing dataset naming is ticker-based.
        if ticker in seen:
            continue
        seen.add(ticker)
        if ticker in results and results[ticker].get("status") == "ok":
            continue
        if attempted >= max_requests:
            break
        if mic not in eligible_mics:
            results[ticker] = {"status": "unsupported_mic", "mic": mic}
            _write_manifest(manifest, payload)
            continue
        attempted += 1
        try:
            dataset_id = DatasetId(f"market.ohlcv.daily.{ticker.lower()}")
            response = ingestor.ingest(
                daily_request(ticker, mic, session, session),
                known_at,
                dataset_id,
            )
            results[ticker] = {
                "status": "ok",
                "mic": mic,
                "security_id": member["security_id"],
                "dataset_id": str(response.metadata.dataset_id),
                "version": str(response.metadata.version),
                "rows": response.metadata.row_count,
            }
            if on_progress is not None:
                on_progress(f"OK {ticker}:{mic} -> {response.metadata.version}")
        except Exception as exc:
            # Keep a failed attempt visible and resumable rather than aborting
            # a long request batch. Never write invented dataset versions.
            results[ticker] = {
                "status": "error",
                "mic": mic,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if on_progress is not None:
                on_progress(f"FAILED {ticker}:{mic}: {exc}")
        _write_manifest(manifest, payload)
    if not manifest.exists():
        _write_manifest(manifest, payload)
    return payload
