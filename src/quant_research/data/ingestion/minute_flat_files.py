"""Verified market-wide Massive minute aggregate downloads."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from quant_research.data.ingestion import massive_flat_files as flat


def minute_key(session: date) -> str:
    return f"us_stocks_sip/minute_aggs_v1/{session:%Y/%m/%Y-%m-%d}.csv.gz"


def download_minute_flat_file(
    *, client: flat.FlatFileClient, session: date, output_root: Path, manifest_path: Path
) -> dict[str, Any]:
    """Download using existing integrity checks and immutable revision manifests."""
    # Reuse the existing downloader; its key construction is the only difference.
    # A scoped override is avoided to prevent mutable process-global behavior.
    return _download(client, session, output_root, manifest_path)


def _download(
    client: flat.FlatFileClient, session: date, output_root: Path, manifest_path: Path
) -> dict[str, Any]:
    import gzip
    import hashlib
    import json
    import os
    from uuid import uuid4

    key = minute_key(session)
    bucket = "flatfiles"
    initial = client.head_object(Bucket=bucket, Key=key)
    identity = (
        str(initial.get("ETag", "")),
        str(initial.get("LastModified", "")),
        int(initial["ContentLength"]),
    )
    if identity[2] <= 0:
        raise ValueError("Empty remote minute file")
    payload: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"format": 1, "bucket": bucket, "key": key, "revisions": []}
    )
    if (payload.get("format"), payload.get("bucket"), payload.get("key")) != (1, bucket, key):
        raise ValueError("Minute flat-file manifest mismatch")

    def verify(path: Path) -> tuple[str, int, int]:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        with gzip.open(path, "rb") as stream:
            rows = sum(1 for _ in stream) - 1
        if rows <= 0:
            raise ValueError("Minute file has no data")
        return digest.hexdigest(), path.stat().st_size, rows

    for revision in payload["revisions"]:
        if (revision["etag"], revision["last_modified"], revision["source_bytes"]) == identity:
            path = Path(revision["path"])
            if not path.is_file() or verify(path) != (
                revision["sha256"],
                revision["bytes"],
                revision["rows"],
            ):
                raise ValueError("Pinned minute flat file missing/corrupt")
            return revision
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{session}.{uuid4().hex}.download"
    try:
        client.download_file(bucket, key, str(temporary))
        latest = client.head_object(Bucket=bucket, Key=key)
        if (
            str(latest.get("ETag", "")),
            str(latest.get("LastModified", "")),
            int(latest["ContentLength"]),
        ) != identity:
            raise flat.SourceChangedError("Minute flat file changed during download")
        sha, size, rows = verify(temporary)
        if size != identity[2]:
            raise ValueError("Minute download size mismatch")
        target = output_root / f"{session}.{sha[:20]}.csv.gz"
        if target.exists():
            if verify(target)[0] != sha:
                raise ValueError("Content-addressed cache collision")
        else:
            os.replace(temporary, target)
        revision = {
            "session": session.isoformat(),
            "path": str(target.resolve()),
            "sha256": sha,
            "bytes": size,
            "rows": rows,
            "source_bytes": identity[2],
            "etag": identity[0],
            "last_modified": identity[1],
        }
        payload["revisions"].append(revision)
        flat._write_manifest(manifest_path, payload)
        return revision
    finally:
        temporary.unlink(missing_ok=True)
