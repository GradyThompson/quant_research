"""Download immutable, checked local copies of Massive market-wide daily flat files."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


class FlatFileClient(Protocol):
    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...


class SourceChangedError(RuntimeError):
    """The upstream object changed during download; retry a fresh invocation."""


def daily_key(session: date) -> str:
    """Exact Massive S3 object key for an aggregate trading day."""
    return f"us_stocks_sip/day_aggs_v1/{session:%Y/%m/%Y-%m-%d}.csv.gz"


def _digest_and_check_gzip(path: Path) -> tuple[str, int, int]:
    """Check gzip integrity including trailer CRC while hashing stored bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    lines = 0
    with gzip.open(path, "rb") as stream:
        for _ in stream:
            lines += 1
    if lines < 2:
        raise ValueError("Aggregate file has no data records")
    return digest.hexdigest(), path.stat().st_size, lines - 1


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def download_daily_flat_file(
    *,
    client: FlatFileClient,
    session: date,
    output_root: Path,
    manifest_path: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cache and audit a complete compressed daily file, never overwrite a pinned copy.

    A same-day upstream revision is saved under a new content-hash filename.
    Manifest history retains every observed revision and its S3 metadata.
    """
    key = daily_key(session)
    bucket = "flatfiles"
    initial = client.head_object(Bucket=bucket, Key=key)
    etag = str(initial.get("ETag", ""))
    changed_at = str(initial.get("LastModified", ""))
    size = int(initial["ContentLength"])
    if size < 1:
        raise ValueError("Remote flat file is empty")
    if manifest_path.exists():
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("format") != 1
            or payload.get("bucket") != bucket
            or payload.get("key") != key
        ):
            raise ValueError("Manifest key mismatch: use a date-specific manifest")
    else:
        payload = {"format": 1, "bucket": bucket, "key": key, "revisions": []}
    revisions = payload["revisions"]
    if not isinstance(revisions, list):
        raise ValueError("Malformed manifest revisions")
    for entry in revisions:
        if (
            entry["etag"] == etag
            and entry["last_modified"] == changed_at
            and entry["source_bytes"] == size
        ):
            cached = Path(entry["path"])
            if cached.is_file():
                sha, bytes_count, rows = _digest_and_check_gzip(cached)
                if (sha, bytes_count, rows) == (entry["sha256"], entry["bytes"], entry["rows"]):
                    if on_progress is not None:
                        on_progress(f"CACHED {session}: {cached}")
                    return entry
            raise ValueError(
                "Pinned local flat file missing or corrupted; manual recovery required"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{session.isoformat()}.{uuid4().hex}.download"
    try:
        client.download_file(bucket, key, str(temporary))
        latest = client.head_object(Bucket=bucket, Key=key)
        if (
            str(latest.get("ETag", "")) != etag
            or str(latest.get("LastModified", "")) != changed_at
            or int(latest["ContentLength"]) != size
        ):
            raise SourceChangedError("Remote object changed during download")
        sha, byte_count, rows = _digest_and_check_gzip(temporary)
        if byte_count != size:
            raise ValueError("Downloaded byte count differs from remote metadata")
        destination = output_root / f"{session.isoformat()}.{sha[:20]}.csv.gz"
        if destination.exists():
            existing_sha, _, _ = _digest_and_check_gzip(destination)
            if existing_sha != sha:
                raise ValueError("Content-addressed cache collision")
        else:
            os.replace(temporary, destination)
        entry = {
            "session": session.isoformat(),
            "path": str(destination.resolve()),
            "sha256": sha,
            "bytes": byte_count,
            "rows": rows,
            "source_bytes": size,
            "etag": etag,
            "last_modified": changed_at,
        }
        revisions.append(entry)
        _write_manifest(manifest_path, payload)
        if on_progress is not None:
            on_progress(f"DOWNLOADED {session}: {rows} rows, sha256={sha}, path={destination}")
        return entry
    finally:
        temporary.unlink(missing_ok=True)
