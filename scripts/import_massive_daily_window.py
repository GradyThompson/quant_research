"""Download and import a pinned window of Massive market-wide daily aggregates."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.ingestion.massive_daily_import import import_verified_daily_file
from quant_research.data.ingestion.massive_daily_window import process_daily_window
from quant_research.data.ingestion.massive_flat_files import download_daily_flat_file
from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import PROJECT_ROOT, resolve_project_path
from quant_research.security_master.persistence import SecurityMasterRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--security-manifest",
        type=Path,
        default=Path("data/dev/security_master_window_manifest.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--flat-file-root",
        type=Path,
        default=Path("data/dev/massive_flat_files"),
    )
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()
    if args.max_sessions < 1:
        parser.error("--max-sessions must be positive")

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.aws_profile and (os.getenv("ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")):
        parser.error("Do not combine --aws-profile with environment credentials")
    if not args.aws_profile:
        # Map existing Massive names to the standard Boto3 credential chain.
        if not os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("ACCESS_KEY_ID"):
            os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
        if not os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("MASSIVE_API_KEY"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MASSIVE_API_KEY"]
    try:
        import boto3
    except ImportError:
        parser.error("boto3 required: uv add boto3")
    session = boto3.Session(profile_name=args.aws_profile)
    client = session.client("s3", endpoint_url="https://files.massive.com", region_name="us-east-1")

    store_root = resolve_project_path(args.store_root)
    root = resolve_project_path(args.flat_file_root)
    store = ParquetDatasetStore(store_root)
    repository = SecurityMasterRepository(store)
    manifest_root = root / "manifests"
    import_root = root / "imports"

    def process(day: date, snapshot_text: str) -> dict[str, object]:
        snapshot = DatasetVersion(UUID(snapshot_text))
        output = import_root / f"{day.isoformat()}.json"
        source_manifest = manifest_root / f"{day.isoformat()}.json"
        # A pre-existing pinned import must be verified, not silently overwritten.
        if output.exists():
            raise ValueError("Import already exists; use the pinned import manifest")
        downloaded = download_daily_flat_file(
            client=client,
            session=day,
            output_root=root / "day_aggs",
            manifest_path=source_manifest,
            on_progress=print,
        )
        resolver = repository.load(snapshot)
        return import_verified_daily_file(
            store=store,
            resolver=resolver,
            session=day,
            source_manifest=source_manifest,
            source_sha256=str(downloaded["sha256"]),
            snapshot_version=snapshot,
            output_manifest=output,
        )

    def validate_existing(record: dict[str, object]) -> bool:
        try:
            dataset = DatasetId(str(record["dataset_id"]))
            version = DatasetVersion(UUID(str(record["dataset_version"])))
            return store.exists(dataset, version)
        except (KeyError, TypeError, ValueError):
            return False

    result = process_daily_window(
        security_manifest=resolve_project_path(args.security_manifest),
        import_root=import_root,
        max_sessions=args.max_sessions,
        process_session=process,
        on_progress=print,
        validate_existing=validate_existing,
    )
    print(f"Summary: {result}")
    print(f"Import manifests: {import_root}")
    print("NOTICE: reference snapshots are retrospective; not verified historical availability")
    print("NOTICE: unadjusted daily bars; corporate actions not incorporated")


if __name__ == "__main__":
    main()
