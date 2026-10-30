"""Research-only first-touch label from one pinned Massive minute session."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quant_research.data.storage import ParquetDatasetStore
from quant_research.labels.pinned_barrier import label_pinned_minute_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/massive_minute_window_receipt.json"),
    )
    parser.add_argument("--entry-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--max-holding-minutes", type=int, default=30)
    args = parser.parse_args()
    result = label_pinned_minute_event(
        store=ParquetDatasetStore(args.store_root),
        receipt_path=args.receipt,
        entry_at=args.entry_at,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
    )
    print(f"Security: {result.security_id}")
    print(f"Dataset: {result.dataset_id}/{result.dataset_version}")
    print(f"Research entry reference (bar close): {result.entry_price:.4f}")
    print(f"Outcome: {result.label.outcome.value}")
    print(f"Observed at: {result.label.observed_at.isoformat()}")
    print(f"Label available at: {result.label.label_available_at.isoformat()}")


if __name__ == "__main__":
    main()
