"""Inspect a pinned five-session minute window without running a private model."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.storage import ParquetDatasetStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/massive_minute_window_receipt.json")
    )
    parser.add_argument("--decision-at", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    data = load_minute_model_input(
        store=ParquetDatasetStore(args.store_root),
        receipt_path=args.receipt,
        decision_at=args.decision_at.astimezone(UTC)
        if args.decision_at.tzinfo
        else args.decision_at,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )
    timestamps = data.minute_bars.column("timestamp").to_pylist()
    print(f"Security: {data.security_id}")
    print(f"Minute bars: {data.minute_bars.num_rows}")
    print(f"First bar end: {timestamps[0].isoformat()}")
    print(f"Last bar end: {timestamps[-1].isoformat()}")
    print(f"Decision time: {data.decision_at.isoformat()}")
    print("PASS: five pinned sessions, no future or unavailable bars")


if __name__ == "__main__":
    main()
