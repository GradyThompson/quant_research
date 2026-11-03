"""Resume historical minute receipt using Massive market-wide minute flat files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.massive_s3 import create_massive_s3_client
from quant_research.data.ingestion.minute_bulk_import import (
    ingest_bulk_session,
    validate_bulk_receipt,
)
from quant_research.data.ingestion.minute_flat_files import download_minute_flat_file
from quant_research.data.ingestion.minute_pilot_resolver import resolve_historical_member
from quant_research.data.ingestion.minute_pilot_window import _write_json_atomic
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/dev/minute_training_plan_20250113.json")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_training_receipt_20250113.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/dev/massive_flat_files/minute_aggs")
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/minute_manifests")
    )
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--publication-lag-seconds", type=int, default=60)
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()
    if args.max_sessions < 1 or args.publication_lag_seconds < 1:
        parser.error("max-sessions and publication-lag-seconds must be positive")
    plan_path = resolve_project_path(args.plan)
    receipt_path = resolve_project_path(args.receipt)
    plan, receipt = validate_bulk_receipt(plan_path, receipt_path, args.publication_lag_seconds)
    store = ParquetDatasetStore(resolve_project_path(args.store_root))
    sessions = [date.fromisoformat(s) for s in plan["initialization_sessions"]]
    sessions.append(date.fromisoformat(plan["evaluation_session"]))
    pending = [
        day
        for day in sessions
        if any(f"{m['ticker']}:{m['mic']}:{day}" not in receipt["entries"] for m in plan["members"])
    ]
    if not pending:
        print("All security-session datasets completed")
        return
    try:
        client = create_massive_s3_client(aws_profile=args.aws_profile)
    except ValueError as exc:
        parser.error(str(exc))
    repository = SecurityMasterRepository(store)
    pinned = {
        date.fromisoformat(item["session"]): item["snapshot_version"]
        for item in plan["source_daily_datasets"]
    }
    pinned[date.fromisoformat(plan["evaluation_session"])] = plan[
        "security_master_snapshot_version"
    ]
    total = 0
    for day in pending[: args.max_sessions]:
        if day not in pinned:
            raise ValueError(f"No pinned reference snapshot for {day}")
        resolver = repository.load(DatasetVersion(UUID(pinned[day])))
        opening = xcals.get_calendar("XNYS").session_open(pd.Timestamp(day)).to_pydatetime()
        for member in plan["members"]:
            resolve_historical_member(
                resolver, member, valid_at=opening, known_at=datetime.now(UTC)
            )
        revision = download_minute_flat_file(
            client=client,
            session=day,
            output_root=resolve_project_path(args.cache_root),
            manifest_path=resolve_project_path(args.manifest_root) / f"{day}.json",
        )
        completed = ingest_bulk_session(
            plan=plan,
            receipt=receipt,
            receipt_path=receipt_path,
            store=store,
            session=day,
            source=Path(revision["path"]),
            source_sha256=revision["sha256"],
            publication_lag_seconds=args.publication_lag_seconds,
        )
        total += completed
        print(f"OK {day}: {completed} newly completed security datasets")
    receipt["completed"] = len(receipt["entries"])
    receipt["total"] = plan["planned_requests"]
    receipt["remaining"] = receipt["total"] - receipt["completed"]
    _write_json_atomic(receipt_path, receipt)
    summary = {
        "new": total,
        "completed": receipt["completed"],
        "remaining": receipt["remaining"],
    }
    print(f"Summary: {json.dumps(summary)}")
    print("NOTICE: retrospective identities; flat-file availability timing unverified")


if __name__ == "__main__":
    main()
