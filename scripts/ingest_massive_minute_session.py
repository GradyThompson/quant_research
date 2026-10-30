"""Ingest one historical Massive minute session into immutable Parquet."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetId, DatasetVersion, TimeRange
from quant_research.data.ingestion.minute import MinuteBarIngestor
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
    parser.add_argument("--date", type=date.fromisoformat, default=date(2025, 1, 8))
    parser.add_argument("--publication-lag-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.publication_lag_seconds <= 0:
        parser.error("publication lag must be positive")
    if not args.ticker.isalpha() or not args.ticker.isupper():
        parser.error("ticker must consist of uppercase ASCII letters")
    session = pd.Timestamp(args.date)
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(session):
        parser.error(f"Not a regular US equity trading session: {args.date}")
    start = calendar.session_open(session).to_pydatetime()
    close = calendar.session_close(session).to_pydatetime()
    request = BarRequest(
        instrument=ProviderInstrument(args.ticker, InstrumentKind.TICKER, args.mic),
        frequency=BarFrequency.MINUTE,
        period=TimeRange(start, close + timedelta(microseconds=1)),
    )
    store = ParquetDatasetStore(args.store_root)
    resolver = SecurityMasterRepository(store).load(DatasetVersion(args.snapshot_version))
    ingestor = MinuteBarIngestor(
        provider=MassiveMinuteProvider(),
        resolver=resolver,
        store=store,
        publication_lag=timedelta(seconds=args.publication_lag_seconds),
    )
    dataset_id = DatasetId(f"market.ohlcv.minute.{args.ticker.lower()}")
    result = ingestor.ingest(request, known_at=datetime.now(UTC), dataset_id=dataset_id)
    table = store.read(dataset_id, result.metadata.version)
    if table.num_rows != result.metadata.row_count:
        raise RuntimeError("Readback row count mismatch")
    print(
        json.dumps(
            {
                "ticker": args.ticker,
                "mic": args.mic,
                "session": args.date.isoformat(),
                "security_id": str(result.security_id),
                "dataset_id": str(dataset_id),
                "version": str(result.metadata.version),
                "rows": table.num_rows,
                "first_bar_end": table.column("timestamp")[0].as_py().isoformat(),
                "last_bar_end": table.column("timestamp")[-1].as_py().isoformat(),
                "publication_lag_seconds": args.publication_lag_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
