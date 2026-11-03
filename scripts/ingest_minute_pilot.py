"""Download and validate a bounded batch of pilot minute security-sessions."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

from quant_research.data.ingestion.minute import MinuteBarIngestor
from quant_research.data.ingestion.minute_pilot_resolver import build_minute_pilot_resolver
from quant_research.data.ingestion.minute_pilot_window import ingest_minute_pilot, load_plan
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("data/dev/minute_pilot_20250113.json"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_pilot_receipt_20250113.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument("--publication-lag-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.max_requests < 1 or args.publication_lag_seconds < 1:
        parser.error("max-requests and publication-lag-seconds must be positive")
    load_dotenv()
    plan_path = resolve_project_path(args.plan)
    plan = load_plan(plan_path)
    store = ParquetDatasetStore(resolve_project_path(args.store_root))
    resolver = build_minute_pilot_resolver(plan, SecurityMasterRepository(store))
    ingestor = MinuteBarIngestor(
        provider=MassiveMinuteProvider(),
        resolver=resolver,
        store=store,
        publication_lag=timedelta(seconds=args.publication_lag_seconds),
    )
    result = ingest_minute_pilot(
        plan_path=plan_path,
        receipt_path=resolve_project_path(args.receipt),
        store=store,
        ingestor=ingestor,
        max_requests=args.max_requests,
        publication_lag_seconds=args.publication_lag_seconds,
    )
    print(f"Summary: {result}")
    print("NOTICE: retrospective security identities; bar publication timing assumed")


if __name__ == "__main__":
    main()
