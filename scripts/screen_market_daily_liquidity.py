"""Screen a candidate universe against pinned market-wide daily import manifests."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path
from quant_research.security_master.market_liquidity import screen_market_daily_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/dev/universe_20250113.json"))
    parser.add_argument(
        "--import-root", type=Path, default=Path("data/dev/massive_flat_files/imports")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/dev/liquid_universe_market_20250113.json")
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-sessions", type=int, default=18)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000.0)
    parser.add_argument("--max-securities", type=int, default=500)
    args = parser.parse_args()
    if args.lookback < 1 or not 1 <= args.min_sessions <= args.lookback:
        parser.error("Invalid lookback/min-sessions")
    cal = xcals.get_calendar("XNYS")
    target = pd.Timestamp(args.as_of)
    if not cal.is_session(target):
        parser.error("Evaluation date must be an exchange trading session")
    index = cal.sessions.get_loc(target)
    if not isinstance(index, int) or index < args.lookback:
        parser.error("Insufficient calendar history")
    dates = tuple(item.date() for item in cal.sessions[index - args.lookback : index])
    universe_path = resolve_project_path(args.universe)
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    if universe.get("valid_date") != args.as_of.isoformat():
        parser.error("Universe valid_date must equal evaluation date")
    members = universe["members"]
    store = ParquetDatasetStore(resolve_project_path(args.store_root))
    source_root = resolve_project_path(args.import_root)
    tables = {}
    source_pins = []
    for day in dates:
        manifest_path = source_root / f"{day.isoformat()}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing import manifest: {manifest_path}")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_id = DatasetId(f"market.ohlcv.daily.us.s{day:%Y%m%d}")
        if record.get("session") != day.isoformat() or record.get("dataset_id") != str(dataset_id):
            raise ValueError(f"Wrong session or dataset in {manifest_path}")
        if record.get("source_adjusted") is not False:
            raise ValueError(f"Expected unadjusted source data: {day}")
        version = DatasetVersion(UUID(record["dataset_version"]))
        if not store.exists(dataset_id, version):
            raise ValueError(f"Pinned daily dataset missing: {dataset_id} / {version}")
        tables[day] = store.read(dataset_id, version)
        source_pins.append(
            {
                "session": day.isoformat(),
                "dataset_id": str(dataset_id),
                "dataset_version": str(version),
                "source_sha256": record["sha256"],
                "snapshot_version": record["snapshot_version"],
            }
        )
    selected, excluded = screen_market_daily_tables(
        members=members,
        daily_tables=tables,
        evaluation_date=args.as_of,
        min_sessions=args.min_sessions,
        min_price=args.min_price,
        min_dollar_volume=args.min_dollar_volume,
        max_securities=args.max_securities,
    )
    result = {
        "schema_version": 1,
        "as_of": args.as_of.isoformat(),
        "lookback_start": dates[0].isoformat(),
        "lookback_end": dates[-1].isoformat(),
        "source_universe": str(universe_path),
        "security_master_snapshot_version": universe.get("snapshot_version"),
        "reference_data_point_in_time_verified": False,
        "daily_data_publication_time_verified": False,
        "daily_version_selection": "pinned_import_manifests",
        "source_daily_datasets": source_pins,
        "criteria": {
            "lookback": args.lookback,
            "min_sessions": args.min_sessions,
            "min_price": args.min_price,
            "min_dollar_volume": args.min_dollar_volume,
            "max_securities": args.max_securities,
        },
        "candidates_checked": len(members),
        "included": len(selected),
        "excluded_reasons": excluded,
        "members": selected,
    }
    output = resolve_project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite screening result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Candidates screened: {len(members)}")
    print(f"Eligible selected: {len(selected)}")
    print(f"Excluded: {excluded}")
    print(f"Lookback: {dates[0]} through {dates[-1]} (evaluation date excluded)")
    print(f"Output: {output}")
    print("NOTICE: retrospective reference data, not verified historically available")
    print("NOTICE: raw daily prices are unadjusted; corporate actions not incorporated")


if __name__ == "__main__":
    main()
