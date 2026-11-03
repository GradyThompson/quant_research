"""Screen a saved security universe using pinned historical daily dataset versions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetId
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.liquidity import screen_daily_bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/dev/universe_20250113.json"))
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/dev/liquid_universe_20250113.json")
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-sessions", type=int, default=18)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000.0)
    parser.add_argument("--max-securities", type=int, default=500)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional pilot cap; sorted by ticker, not liquidity",
    )
    args = parser.parse_args()
    if (
        args.lookback < 1
        or args.max_securities < 1
        or (args.max_candidates is not None and args.max_candidates < 1)
    ):
        parser.error("Lookback and candidate/security limits must be positive")
    calendar = xcals.get_calendar("XNYS")
    target = pd.Timestamp(args.as_of)
    if not calendar.is_session(target):
        parser.error(f"Not an exchange session: {args.as_of}")
    location = calendar.sessions.get_loc(target)
    if not isinstance(location, int) or location < args.lookback:
        parser.error("Not enough calendar sessions before as-of date")
    sessions = tuple(item.date() for item in calendar.sessions[location - args.lookback : location])
    input_path = args.universe if args.universe.is_absolute() else PROJECT_ROOT / args.universe
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("valid_date") != args.as_of.isoformat():
        parser.error("Universe valid_date must equal --as-of")
    members = payload["members"]
    if args.max_candidates is not None:
        members = members[: args.max_candidates]
    store_root = (
        args.store_root if args.store_root.is_absolute() else PROJECT_ROOT / args.store_root
    )
    store = ParquetDatasetStore(store_root)
    excluded: Counter[str] = Counter()
    included: list[dict[str, object]] = []
    for member in members:
        ticker = member["ticker"]
        identifier = DatasetId(f"market.ohlcv.daily.{ticker.lower()}")
        versions = store.list_versions(identifier)
        if not versions:
            excluded["missing_daily_dataset"] += 1
            continue
        latest = versions[-1]
        table = store.read(identifier, latest.version)
        # Reject an accidental ticker reuse or an incorrect security mapping.
        if "security_id" not in table.column_names or set(
            table.column("security_id").to_pylist()
        ) != {member["security_id"]}:
            excluded["security_id_mismatch"] += 1
            continue
        stats = screen_daily_bars(
            table,
            evaluation_date=args.as_of,
            expected_sessions=sessions,
            min_price=args.min_price,
            min_mean_dollar_volume=args.min_dollar_volume,
            min_valid_sessions=args.min_sessions,
        )
        if stats.reason is not None:
            excluded[stats.reason] += 1
            continue
        included.append(
            {
                **member,
                "reference_price": stats.last_close,
                "mean_dollar_volume": stats.mean_dollar_volume,
                "valid_sessions": stats.valid_sessions,
                "daily_dataset_id": str(identifier),
                "daily_dataset_version": str(latest.version),
            }
        )
    included.sort(key=lambda row: (-float(row["mean_dollar_volume"]), str(row["ticker"])))
    selected = included[: args.max_securities]
    excluded["outside_top_n"] += len(included) - len(selected)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite screening output: {output}")
    result = {
        "schema_version": 1,
        "as_of": args.as_of.isoformat(),
        "lookback_start": sessions[0].isoformat(),
        "lookback_end": sessions[-1].isoformat(),
        "source_universe": str(input_path),
        "security_master_snapshot_version": payload.get("snapshot_version"),
        "reference_data_point_in_time_verified": False,
        "daily_data_publication_time_verified": False,
        "daily_version_selection": "latest_stored_at_screen_time",
        "criteria": {
            "lookback": args.lookback,
            "min_sessions": args.min_sessions,
            "min_price": args.min_price,
            "min_dollar_volume": args.min_dollar_volume,
            "max_securities": args.max_securities,
        },
        "candidates_checked": len(members),
        "included": len(selected),
        "excluded_reasons": dict(sorted(excluded.items())),
        "members": selected,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Candidates screened: {len(members)}")
    print(f"Eligible selected: {len(selected)}")
    print(f"Excluded: {dict(sorted(excluded.items()))}")
    print(f"Lookback: {sessions[0]} through {sessions[-1]} (evaluation date excluded)")
    print(f"Output: {output}")
    print(
        "NOTICE: dataset versions pinned in output; latest stored version chosen at screening time"
    )
    print(
        "NOTICE: retrospective reference data and daily publication times not verified point-in-time"
    )


if __name__ == "__main__":
    main()
