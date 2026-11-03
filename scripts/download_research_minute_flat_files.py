"""Download a bounded set of Massive minute flat files for P4.4 research.

Resumable, content-addressed, and restricted to development days; excludes
original final-holdout and original evaluation dates before any S3 request.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.ingestion.massive_s3 import create_massive_s3_client
from quant_research.data.ingestion.minute_flat_files import download_minute_flat_file
from quant_research.model.opening_range_history import protected_research_days
from quant_research.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument(
        "--original-receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/dev/massive_flat_files/minute_aggs")
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/minute_manifests")
    )
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()
    if args.start > args.end or args.max_sessions < 1:
        parser.error("invalid range or max-sessions")
    original = json.loads(resolve_project_path(args.original_receipt).read_text(encoding="utf-8"))
    protected = protected_research_days(original)
    calendar = xcals.get_calendar("XNYS")
    sessions = [
        stamp.date()
        for stamp in calendar.sessions_in_range(pd.Timestamp(args.start), pd.Timestamp(args.end))
        if stamp.date() not in protected
    ]
    cache = resolve_project_path(args.cache_root)
    manifests = resolve_project_path(args.manifest_root)
    # Resume by checking whether a dated manifest already exists. An existing
    # manifest is NEVER silently overwritten (research can verify its hash).
    pending = [day for day in sessions if not (manifests / f"{day}.json").exists()]
    if not pending:
        print("All permitted dates already have local manifests")
        return
    try:
        client = create_massive_s3_client(aws_profile=args.aws_profile)
    except ValueError as exc:
        parser.error(str(exc))
    for day in pending[: args.max_sessions]:
        revision = download_minute_flat_file(
            client=client,
            session=day,
            output_root=cache,
            manifest_path=manifests / f"{day}.json",
        )
        print(f"OK {day}: SHA256={revision['sha256']}")
    print(f"Downloaded {min(len(pending), args.max_sessions)} of {len(pending)} pending dates")
    print("NOTICE: historical flat-file publication times are not directly verified")


if __name__ == "__main__":
    main()
