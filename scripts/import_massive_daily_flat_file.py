"""Import one pinned Massive daily flat file into versioned market-wide OHLCV."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetVersion
from quant_research.data.ingestion.massive_daily_import import import_verified_daily_file
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=date.fromisoformat, required=True)
    parser.add_argument("--snapshot-version", type=UUID, required=True)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--sha256", required=True, help="Exact source revision SHA-256")
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    session_key = args.session.isoformat()
    root = resolve_project_path(args.store_root)
    source_manifest = resolve_project_path(
        args.source_manifest or root / "massive_flat_files" / "manifests" / f"{session_key}.json"
    )
    output_manifest = resolve_project_path(
        args.output_manifest or root / "massive_flat_files" / "imports" / f"{session_key}.json"
    )
    store = ParquetDatasetStore(root)
    resolver = SecurityMasterRepository(store).load(DatasetVersion(args.snapshot_version))
    result = import_verified_daily_file(
        store=store,
        resolver=resolver,
        session=args.session,
        source_manifest=source_manifest,
        source_sha256=args.sha256,
        snapshot_version=DatasetVersion(args.snapshot_version),
        output_manifest=output_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Import manifest: {output_manifest}")
    print("NOTICE: retrospective reference mappings, not verified historical knowledge")


if __name__ == "__main__":
    main()
