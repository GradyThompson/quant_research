"""Ingest a small US-equity daily dataset using a pinned security master."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.daily import DailyBarIngestor
from quant_research.data.ingestion.massive_pilot import ingest_pilot, write_receipt
from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.storage import ParquetDatasetStore
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--snapshot-version", type=UUID, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL:XNAS", "MSFT:XNAS", "NVDA:XNAS", "JPM:XNYS", "XOM:XNYS"],
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/massive_daily_pilot_receipt.json")
    )
    args = parser.parse_args()
    targets: list[tuple[str, str]] = []
    for item in args.symbols:
        if item.count(":") != 1:
            parser.error(f"Invalid symbol, expected TICKER:MIC: {item}")
        ticker, mic = item.split(":")
        if not ticker or not mic:
            parser.error(f"Invalid symbol: {item}")
        targets.append((ticker, mic))

    store = ParquetDatasetStore(args.store_root)
    version = DatasetVersion(args.snapshot_version)
    resolver = SecurityMasterRepository(store).load(version)
    ingestor = DailyBarIngestor(provider=MassiveDailyProvider(), resolver=resolver, store=store)
    receipts = ingest_pilot(
        ingestor,
        targets=tuple(targets),
        start=args.start,
        end=args.end,
        known_at=datetime.now(UTC),
    )
    write_receipt(args.receipt, snapshot_version=version, receipts=receipts)
    for receipt in receipts:
        print(
            f"{receipt['ticker']}:{receipt['mic']} {receipt['rows']} bars -> {receipt['dataset_id']} @ {receipt['version']}"
        )
    print(f"Receipt: {args.receipt}")


if __name__ == "__main__":
    main()
