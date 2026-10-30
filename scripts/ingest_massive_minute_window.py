"""Ingest five completed US-equity sessions into separately pinned minute datasets."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion, TimeRange
from quant_research.data.ingestion.minute import MinuteBarIngestor
from quant_research.data.ingestion.minute_window import previous_sessions, session_dataset_id
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.providers.models import BarRequest, InstrumentKind, ProviderInstrument
from quant_research.data.schemas import BarFrequency
from quant_research.data.storage import ParquetDatasetStore
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--snapshot-version", type=UUID, required=True)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--mic", choices=("XNAS", "XNYS"), default="XNAS")
    parser.add_argument("--end-session", type=date.fromisoformat, default=date(2025, 1, 10))
    parser.add_argument("--publication-lag-seconds", type=int, default=60)
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/massive_minute_window_receipt.json")
    )
    args = parser.parse_args()
    if args.publication_lag_seconds <= 0:
        parser.error("publication lag must be positive")
    if not args.ticker.isascii() or not args.ticker.isalpha() or not args.ticker.isupper():
        parser.error("ticker must be uppercase ASCII letters")
    try:
        sessions = previous_sessions(end_session=args.end_session)
    except ValueError as exc:
        parser.error(str(exc))
    if args.receipt.exists():
        parser.error(f"Receipt already exists; refusing to overwrite: {args.receipt}")

    calendar = xcals.get_calendar("XNYS")
    store = ParquetDatasetStore(args.store_root)
    resolver = SecurityMasterRepository(store).load(DatasetVersion(args.snapshot_version))
    ingestor = MinuteBarIngestor(
        provider=MassiveMinuteProvider(),
        resolver=resolver,
        store=store,
        publication_lag=timedelta(seconds=args.publication_lag_seconds),
    )
    entries: list[dict[str, object]] = []
    for day in sessions:
        session = pd.Timestamp(day)
        opening = calendar.session_open(session).to_pydatetime()
        closing = calendar.session_close(session).to_pydatetime()
        request = BarRequest(
            instrument=ProviderInstrument(args.ticker, InstrumentKind.TICKER, args.mic),
            frequency=BarFrequency.MINUTE,
            period=TimeRange(opening, closing + timedelta(microseconds=1)),
        )
        dataset_id = session_dataset_id(args.ticker, day)
        result = ingestor.ingest(request, known_at=datetime.now(UTC), dataset_id=dataset_id)
        saved = store.read(dataset_id, result.metadata.version)
        if saved.num_rows != result.metadata.row_count:
            raise RuntimeError(f"Readback row count mismatch for {day}")
        entry: dict[str, object] = {
            "session": day.isoformat(),
            "dataset_id": str(dataset_id),
            "version": str(result.metadata.version),
            "security_id": str(result.security_id),
            "rows": saved.num_rows,
        }
        entries.append(entry)
        print(f"{day}: {saved.num_rows} bars -> {dataset_id}/{result.metadata.version}")

    if len({entry["security_id"] for entry in entries}) != 1:
        raise RuntimeError("Security resolution changed across the five sessions")
    receipt = {
        "ticker": args.ticker,
        "mic": args.mic,
        "snapshot_version": str(args.snapshot_version),
        "publication_lag_seconds": args.publication_lag_seconds,
        "sessions": entries,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Saved receipt to {args.receipt}")


if __name__ == "__main__":
    main()
