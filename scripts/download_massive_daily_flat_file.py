"""Download and verify a single market-wide Massive day-aggregates file."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from quant_research.data.ingestion.massive_flat_files import download_daily_flat_file
from quant_research.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/dev/massive_flat_files/day_aggs")
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/manifests")
    )
    parser.add_argument(
        "--aws-profile", default=None, help="Optional AWS shared-credentials profile"
    )
    args = parser.parse_args()
    try:
        import boto3
    except ImportError as exc:
        parser.error("boto3 is required: run 'uv add boto3'")
        raise AssertionError from exc
    # boto3 reads AWS credentials/profile automatically. Never print credentials.
    client_kwargs: dict[str, str] = {
        "endpoint_url": "https://files.massive.com",
        "region_name": "us-east-1",
    }
    if args.aws_profile is not None and any(
        os.environ.get(key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    ):
        parser.error("Do not mix --aws-profile with AWS environment credentials")
    session = boto3.Session(profile_name=args.aws_profile)
    client = session.client("s3", **client_kwargs)
    result = download_daily_flat_file(
        client=client,
        session=args.session,
        output_root=resolve_project_path(args.output_root),
        manifest_path=resolve_project_path(args.manifest_root) / f"{args.session.isoformat()}.json",
        on_progress=print,
    )
    print(f"Verified: {result['path']}")
    print(f"Rows: {result['rows']}; sha256: {result['sha256']}")
    print("NOTICE: unadjusted market-wide prices; not yet mapped to internal security IDs")
    print("NOTICE: source object may be revised; manifest records observed versions")


if __name__ == "__main__":
    main()
