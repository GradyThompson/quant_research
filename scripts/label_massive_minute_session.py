"""Generate research barrier labels from one pinned Massive minute session."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path

from quant_research.data.storage import ParquetDatasetStore
from quant_research.labels.session_barriers import label_pinned_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/massive_minute_window_receipt.json"),
    )
    parser.add_argument("--session", type=date.fromisoformat, required=True)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--max-holding-minutes", type=int, default=30)
    parser.add_argument("--stride-minutes", type=int, default=5)
    args = parser.parse_args()

    batch = label_pinned_session(
        store=ParquetDatasetStore(args.store_root),
        receipt_path=args.receipt,
        session=args.session,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
        stride_minutes=args.stride_minutes,
    )
    counts = Counter(event.label.outcome.value for event in batch.events)
    print(f"Security: {batch.security_id}")
    print(f"Pinned dataset: {batch.dataset_id}/{batch.dataset_version}")
    print(f"Labeled events: {len(batch.events)}")
    print(f"Skipped incomplete windows: {batch.skipped_incomplete}")
    for outcome in ("upper_first", "lower_first", "timeout", "ambiguous"):
        print(f"{outcome}: {counts[outcome]}")


if __name__ == "__main__":
    main()
