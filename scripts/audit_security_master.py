"""Audit a pinned historical security-master snapshot before universe construction."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, time
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetVersion
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT
from quant_research.security_master.coverage import audit_security_master
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default="data/dev")
    parser.add_argument(
        "--snapshot-version", required=True, help="Pinned security master manifest UUID"
    )
    parser.add_argument("--valid-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--known-at",
        type=datetime.fromisoformat,
        required=True,
        help="UTC-aware knowledge cutoff; later cutoffs constitute hindsight",
    )
    parser.add_argument("--fail-on-conflicts", action="store_true")
    args = parser.parse_args()
    root = Path(args.store_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    repository = SecurityMasterRepository(ParquetDatasetStore(root))
    version = DatasetVersion(UUID(args.snapshot_version))
    resolver = repository.load(version)
    report = audit_security_master(
        resolver,
        valid_at=datetime.combine(args.valid_date, time.min, UTC),
        known_at=args.known_at,
    )
    print(f"Snapshot: {version}")
    print(f"Valid at: {report.valid_at.isoformat()}")
    print(f"Known at: {report.known_at.isoformat()}")
    print(f"Registered securities: {report.registered_securities}")
    print(f"Actively listed securities: {report.active_securities}")
    print(f"Active listings: {report.active_listings}")
    print(f"Active identifiers: {report.active_identifiers}")
    print(f"Active security types: {report.active_by_type}")
    print(f"Active securities missing share-class FIGI: {len(report.missing_share_class_figi)}")
    print(f"Registered securities without active listing: {len(report.missing_active_listing)}")
    print(f"Securities with multiple active listings: {len(report.multi_listing_securities)}")
    print(f"Ticker/MIC conflicts: {report.ticker_mic_conflicts}")
    print(f"Identifier conflicts: {report.identifier_conflicts}")
    print("NOTE: coverage describes assertions in the snapshot, not the complete market.")
    if args.fail_on_conflicts and report.has_conflicts:
        parser.exit(status=1, message="Security-master conflicts detected.\n")


if __name__ == "__main__":
    main()
