"""Discover historical NYSE-session inventories as separate immutable snapshots."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.historical_window import (
    reconstruct_dated_window,
    session_dates,
)
from quant_research.security_master.massive_discovery import discover_massive_common_stocks
from quant_research.security_master.persistence import SecurityMasterRepository


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/dev/security_master_window_manifest.json")
    )
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--active-only", action="store_true")
    args = parser.parse_args()
    dates = session_dates(args.start, args.end)
    if not dates:
        parser.error("No exchange sessions in requested window")
    if args.max_sessions < 1:
        parser.error("--max-sessions must be positive")

    root = _absolute(args.store_root)
    manifest = _absolute(args.manifest)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        parser.error("MASSIVE_API_KEY is missing")
    client = RESTClient(api_key=api_key)
    repository = SecurityMasterRepository(ParquetDatasetStore(root))

    def discover(day: date) -> tuple[str, dict[str, object]]:
        resolver, report = discover_massive_common_stocks(
            client,
            as_of=day,
            recorded_at=datetime.now(UTC),
            active=True if args.active_only else None,
        )
        snapshot = repository.save(resolver)
        details = asdict(report)
        details["as_of"] = report.as_of.isoformat()
        return str(snapshot.version), details

    payload = reconstruct_dated_window(
        dates=dates,
        manifest_path=manifest,
        store_root=root,
        discover_and_save=discover,
        max_sessions=args.max_sessions,
        on_progress=print,
    )
    results = payload["results"]
    succeeded = sum(result["status"] == "ok" for result in results.values())
    failed = sum(result["status"] == "failed" for result in results.values())
    print(f"Sessions: {len(dates)}; completed: {succeeded}; failed: {failed}")
    print(f"Manifest: {manifest}")
    print("NOTICE: dated inventories are separate; no continuity is inferred")
    print("NOTICE: retrieved information is NOT verified available on historical dates")


if __name__ == "__main__":
    main()
