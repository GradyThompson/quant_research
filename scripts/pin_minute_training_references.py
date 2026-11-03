"""Pin historical security-master snapshots into an expanded minute plan.

Requires a completed dated discovery manifest. Existing minute receipt entries
are preserved, and the plan fingerprint is updated only after all validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.minute_pilot_resolver import resolve_historical_member
from quant_research.data.ingestion.minute_pilot_window import load_plan
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path
from quant_research.security_master.persistence import SecurityMasterRepository


def pin_references(
    plan_path: Path, receipt_path: Path, manifest_path: Path, store_root: Path
) -> int:
    """Validate all dated snapshots, then update plan and receipt with backups."""
    plan = load_plan(plan_path)
    original_plan = plan_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_sha256") != hashlib.sha256(original_plan).hexdigest():
        raise ValueError("Receipt/plan fingerprint mismatch; refusing to change either")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or manifest.get("store_root") != str(store_root.resolve()):
        raise ValueError("Discovery manifest format or store root mismatch")
    sessions = [date.fromisoformat(s) for s in plan["initialization_sessions"]]
    required = {s.isoformat() for s in sessions}
    results = manifest.get("results", {})
    missing = sorted(s for s in required if results.get(s, {}).get("status") != "ok")
    existing = {item["session"]: item for item in plan["source_daily_datasets"]}
    missing = [s for s in missing if s not in existing]
    if missing:
        raise ValueError(f"Missing successful dated reference discovery: {missing}")

    repository = SecurityMasterRepository(ParquetDatasetStore(store_root))
    calendar = xcals.get_calendar("XNYS")
    added = 0
    for session in sessions:
        day = session.isoformat()
        if day in existing:
            version = existing[day]["snapshot_version"]
        else:
            version = results[day]["snapshot_version"]
            added += 1
        resolver = repository.load(DatasetVersion(UUID(version)))
        opening = calendar.session_open(pd.Timestamp(session)).to_pydatetime()
        for member in plan["members"]:
            resolve_historical_member(
                resolver, member, valid_at=opening, known_at=datetime.now(UTC)
            )
        existing[day] = {"session": day, "snapshot_version": version}

    if not added:
        print("All required snapshots are already pinned")
        return 0
    plan["source_daily_datasets"] = [existing[d] for d in sorted(existing)]
    new_plan = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    receipt["plan_sha256"] = hashlib.sha256(new_plan).hexdigest()
    new_receipt = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    backups = [(plan_path, original_plan), (receipt_path, receipt_path.read_bytes())]
    for path, content in backups:
        backup = path.with_suffix(path.suffix + ".before_reference_pin")
        with backup.open("xb") as handle:
            handle.write(content)
    # Both updates validated before writes. Backups permit recovery if interrupted.
    plan_path.write_bytes(new_plan)
    receipt_path.write_bytes(new_receipt)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/dev/minute_training_plan_20250113.json")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_training_receipt_20250113.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dev/minute_training_security_master_manifest.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    args = parser.parse_args()
    added = pin_references(
        resolve_project_path(args.plan),
        resolve_project_path(args.receipt),
        resolve_project_path(args.manifest),
        resolve_project_path(args.store_root),
    )
    print(f"Pinned {added} additional dated reference snapshots")
    print("NOTICE: retrospective identities, not point-in-time verified")


if __name__ == "__main__":
    main()
