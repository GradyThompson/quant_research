"""Build a bounded, versioned pilot security master from Massive reference data."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, date, datetime
from pathlib import Path

from dotenv import load_dotenv
from massive import RESTClient

from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.massive_bootstrap import bootstrap_massive_security_master
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default="data/dev")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL:XNAS", "MSFT:XNAS", "NVDA:XNAS", "JPM:XNYS", "XOM:XNYS"],
    )
    parser.add_argument(
        "--confirm-continuity",
        action="store_true",
        help="I independently verified these listings were continuous for this interval",
    )
    args = parser.parse_args()
    if not args.confirm_continuity:
        parser.error(
            "--confirm-continuity required; two endpoint checks cannot prove historical continuity"
        )
    targets: list[tuple[str, str]] = []
    for item in args.symbols:
        parts = item.split(":")
        if len(parts) != 2 or not all(parts):
            parser.error(f"Invalid ticker:MIC target: {item}")
        targets.append((parts[0], parts[1]))
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        parser.error("MASSIVE_API_KEY is missing")
    client = RESTClient(api_key=api_key)
    resolver = bootstrap_massive_security_master(
        client,
        targets,
        valid_from=args.start,
        valid_through=args.end,
        recorded_at=datetime.now(UTC),
        historical_continuity_confirmed=True,
    )
    root = Path(args.store_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    repo = SecurityMasterRepository(ParquetDatasetStore(root))
    snapshot = repo.save(resolver)
    print(f"Snapshot version: {snapshot.version}")
    print(f"Store root: {root}")
    for listing in resolver.listings:
        print(f"{listing.ticker}:{listing.mic} -> {listing.security_id}")
    print("Recorded assertions use today's knowledge, not historical point-in-time knowledge.")


if __name__ == "__main__":
    main()
