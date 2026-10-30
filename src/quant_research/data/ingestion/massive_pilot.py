"""Bounded Massive daily-ingestion pilot using a pinned security-master snapshot."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol

from quant_research.core import DatasetId, DatasetVersion, TimeRange
from quant_research.data.ingestion.daily import DailyIngestionResult
from quant_research.data.providers.models import BarRequest, InstrumentKind, ProviderInstrument
from quant_research.data.schemas import BarFrequency


class DailyIngestor(Protocol):
    def ingest(
        self, request: BarRequest, known_at: datetime, dataset_id: DatasetId | None = None
    ) -> DailyIngestionResult: ...


def daily_request(ticker: str, mic: str, start: date, end: date) -> BarRequest:
    """Inclusive user dates -> half-open UTC window containing all daily closes."""
    if start > end:
        raise ValueError("start must not exceed end")
    if mic not in {"XNAS", "XNYS"}:
        raise ValueError("Pilot supports XNAS and XNYS only")
    return BarRequest(
        instrument=ProviderInstrument(ticker, InstrumentKind.TICKER, mic),
        frequency=BarFrequency.DAILY,
        period=TimeRange(
            datetime.combine(start, time.min, tzinfo=UTC),
            datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC),
        ),
    )


def ingest_pilot(
    ingestor: DailyIngestor,
    *,
    targets: tuple[tuple[str, str], ...],
    start: date,
    end: date,
    known_at: datetime,
) -> list[dict[str, str | int]]:
    """Persist separate immutable datasets. Stop immediately on any failure."""
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be nonempty and unique")
    if known_at.tzinfo is None or known_at.utcoffset() != timedelta(0):
        raise ValueError("known_at must be UTC aware")
    if datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC) > known_at:
        raise ValueError("Requested end date must precede known_at")
    receipts: list[dict[str, str | int]] = []
    for ticker, mic in targets:
        dataset_id = DatasetId(f"market.ohlcv.daily.{ticker.lower()}")
        result = ingestor.ingest(daily_request(ticker, mic, start, end), known_at, dataset_id)
        receipts.append(
            {
                "ticker": ticker,
                "mic": mic,
                "security_id": str(result.security_id),
                "dataset_id": str(result.metadata.dataset_id),
                "version": str(result.metadata.version),
                "rows": result.metadata.row_count,
            }
        )
    return receipts


def write_receipt(
    path: Path, *, snapshot_version: DatasetVersion, receipts: list[dict[str, str | int]]
) -> None:
    """Write an auditable receipt only after all instruments succeed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Receipt already exists: {path}")
    path.write_text(
        json.dumps(
            {"security_master_snapshot": str(snapshot_version), "datasets": receipts}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
