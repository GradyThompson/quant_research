"""Ingest a bounded number of daily bars using dated security-master snapshots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.daily import DailyBarIngestor
from quant_research.data.ingestion.historical_daily_window import ingest_historical_daily_window
from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.persistence import SecurityMasterRepository


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/dev/universe_20250113.json"))
    parser.add_argument(
        "--window-manifest",
        type=Path,
        default=Path("data/dev/security_master_window_manifest.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/dev/historical_daily_ingestion_manifest.json")
    )
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument(
        "--max-candidates", type=int, default=None, help="Pilot subset only; first N by security ID"
    )
    args = parser.parse_args()
    universe = json.loads(_absolute(args.universe).read_text(encoding="utf-8"))
    window = json.loads(_absolute(args.window_manifest).read_text(encoding="utf-8"))
    if window.get("format") != 1 or not isinstance(window.get("results"), dict):
        parser.error("Invalid security-master window manifest")
    failures = [day for day, result in window["results"].items() if result.get("status") != "ok"]
    if failures or set(window["results"]) != set(window.get("dates", [])):
        parser.error("Security-master window is incomplete or has failures")
    snapshots = {day: result["snapshot_version"] for day, result in window["results"].items()}
    members = sorted(
        universe["members"], key=lambda member: (member["security_id"], member["ticker"])
    )
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            parser.error("--max-candidates must be positive")
        members = members[: args.max_candidates]
    store_root = _absolute(args.store_root)
    if window.get("store_root") != str(store_root.resolve()):
        parser.error("Window manifest points to a different dataset store")
    store = ParquetDatasetStore(store_root)
    repository = SecurityMasterRepository(store)
    provider = MassiveDailyProvider()

    resolvers = {}

    def get_resolver(snapshot: str):
        if snapshot not in resolvers:
            resolvers[snapshot] = repository.load(DatasetVersion(UUID(snapshot)))
        return resolvers[snapshot]

    def make_ingestor(snapshot: str) -> DailyBarIngestor:
        return DailyBarIngestor(provider=provider, resolver=get_resolver(snapshot), store=store)

    def historical_listing(snapshot: str, security_id: str) -> tuple[str, str] | None:
        matches = {
            (listing.ticker, listing.mic)
            for listing in get_resolver(snapshot).listings
            if str(listing.security_id) == security_id
        }
        return next(iter(matches)) if len(matches) == 1 else None

    payload = ingest_historical_daily_window(
        members=members,
        snapshots=snapshots,
        manifest_path=_absolute(args.manifest),
        store_root=store_root,
        make_ingestor=make_ingestor,
        historical_listing=historical_listing,
        known_at=datetime.now(UTC),
        max_requests=args.max_requests,
        on_progress=print,
    )
    counts = Counter(result["status"] for result in payload["results"].values())
    print(
        f"Records: {len(payload['results'])} of {len(members) * len(snapshots)} possible; statuses: {dict(sorted(counts.items()))}"
    )
    print(f"Manifest: {_absolute(args.manifest)}")
    print("NOTICE: per-ticker requests; estimated full workload is large")
    print("NOTICE: session datasets are separate; liquidity-window assembly is a subsequent step")
    print("NOTICE: retrospective reference data is not verified point-in-time")


if __name__ == "__main__":
    main()
