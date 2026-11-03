"""Persist a dated, research-only security inventory from Massive reference data."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.massive_discovery import discover_massive_common_stocks
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--store-root", default="data/dev")
    parser.add_argument("--active-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        parser.error("MASSIVE_API_KEY is missing")
    client = RESTClient(api_key=api_key)
    resolver, report = discover_massive_common_stocks(
        client,
        as_of=args.as_of,
        recorded_at=datetime.now(UTC),
        active=True if args.active_only else None,
    )
    root = Path(args.store_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    snapshot = SecurityMasterRepository(ParquetDatasetStore(root)).save(resolver)
    print(f"Snapshot version: {snapshot.version}")
    print(f"Store root: {root}")
    print(f"Report: {report}")
    print("NOTICE: historical validity is limited to the requested date.")
    print("NOTICE: recorded_at is retrieval time, NOT historical availability.")
    print("NOTICE: this is a dated inventory, not yet a verified historical universe.")


if __name__ == "__main__":
    main()
