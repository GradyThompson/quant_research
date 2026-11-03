"""Plan 60 historical sessions and reuse pinned pilot minute datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_research.data.ingestion.minute_history import expand_minute_history
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-plan", type=Path, default=Path("data/dev/minute_pilot_20250113.json")
    )
    parser.add_argument(
        "--source-receipt", type=Path, default=Path("data/dev/minute_pilot_receipt_20250113.json")
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("data/dev/minute_training_plan_20250113.json")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_training_receipt_20250113.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--history-sessions", type=int, default=60)
    args = parser.parse_args()
    result = expand_minute_history(
        source_plan_path=resolve_project_path(args.source_plan),
        source_receipt_path=resolve_project_path(args.source_receipt),
        output_plan_path=resolve_project_path(args.plan),
        output_receipt_path=resolve_project_path(args.receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        history_sessions=args.history_sessions,
    )
    print(f"Expanded plan: {result}")
    print(f"Plan: {args.plan}; receipt: {args.receipt}")
    print("NOTICE: retrospective security identities and assumed publication lag")


if __name__ == "__main__":
    main()
