"""Export a reproducible candidate universe from one security-master snapshot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from pathlib import Path
from uuid import UUID

from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.storage.models import DatasetVersion
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.persistence import SecurityMasterRepository
from quant_research.security_master.universe import build_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default="data/dev")
    parser.add_argument("--snapshot-version", type=UUID, required=True)
    parser.add_argument("--valid-date", type=date.fromisoformat, required=True)
    parser.add_argument("--known-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/dev/universe_20250113.json"))
    parser.add_argument("--mic", action="append", help="Allowed MIC; repeat to include several")
    parser.add_argument("--allow-missing-figi", action="store_true")
    args = parser.parse_args()
    valid_at = datetime.combine(args.valid_date, time(12), tzinfo=UTC)
    root = Path(args.store_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    repo = SecurityMasterRepository(ParquetDatasetStore(root))
    resolver = repo.load(DatasetVersion(args.snapshot_version))
    universe = build_universe(
        resolver,
        valid_at=valid_at,
        known_at=args.known_at,
        require_share_class_figi=not args.allow_missing_figi,
        allowed_mics=frozenset(args.mic) if args.mic else None,
    )
    payload = {
        "schema_version": 1,
        "snapshot_version": str(args.snapshot_version),
        "valid_date": args.valid_date.isoformat(),
        "known_at": args.known_at.isoformat(),
        "allowed_mics": sorted(args.mic) if args.mic else None,
        "require_share_class_figi": not args.allow_missing_figi,
        "reference_data_point_in_time_verified": False,
        "report": asdict(universe.report),
        "members": [
            {"security_id": str(m.security_id), "ticker": m.ticker, "mic": m.mic}
            for m in universe.members
        ],
    }
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Universe: {len(universe.members)} securities")
    print(f"Report: {universe.report}")
    print(f"Output: {output}")
    print("NOTICE: retrospective vendor data, not verified point-in-time historical availability")


if __name__ == "__main__":
    main()
