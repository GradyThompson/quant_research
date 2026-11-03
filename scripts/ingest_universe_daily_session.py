"""Ingest a bounded historical daily OHLCV session for a saved security universe."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.daily import DailyBarIngestor
from quant_research.data.ingestion.universe_daily import ingest_universe_daily_session
from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.persistence import SecurityMasterRepository


def _path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/dev/universe_20250113.json"))
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--session", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/dev/daily_session_20250113_manifest.json")
    )
    parser.add_argument("--max-requests", type=int, default=10)
    args = parser.parse_args()
    if args.max_requests < 1:
        parser.error("--max-requests must be positive")
    payload = json.loads(_path(args.universe).read_text(encoding="utf-8"))
    members = payload["members"]
    if not isinstance(members, list):
        parser.error("Universe members must be a list")
    snapshot = payload.get("snapshot_version")
    if not isinstance(snapshot, str):
        parser.error("Universe is missing snapshot_version")
    store = ParquetDatasetStore(_path(args.store_root))
    resolver = SecurityMasterRepository(store).load(DatasetVersion(UUID(snapshot)))
    ingestor = DailyBarIngestor(provider=MassiveDailyProvider(), resolver=resolver, store=store)
    manifest = ingest_universe_daily_session(
        ingestor=ingestor,
        members=members,
        session=args.session,
        snapshot_version=snapshot,
        manifest=_path(args.manifest),
        max_requests=args.max_requests,
        known_at=datetime.now(UTC),
        on_progress=print,
    )
    results = manifest["results"]
    print(
        f"Recorded: {len(results)}; successes: {sum(v['status'] == 'ok' for v in results.values())}"
    )
    print(f"Manifest: {_path(args.manifest)}")
    print("NOTICE: this is bounded per-ticker ingestion, not market-wide flat-file ingestion")
    print("NOTICE: historical daily bars were retrieved retrospectively")


if __name__ == "__main__":
    main()
