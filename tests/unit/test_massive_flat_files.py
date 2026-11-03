"""Offline S3 fake for the market-wide flat-file cache."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import pytest

from quant_research.data.ingestion.massive_flat_files import (
    SourceChangedError,
    daily_key,
    download_daily_flat_file,
)


class FakeClient:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.downloads = 0
        self.revision = "v1"
        self.change_during_download = False

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == "flatfiles"
        assert Key == "us_stocks_sip/day_aggs_v1/2025/01/2025-01-10.csv.gz"
        return {
            "ETag": self.revision,
            "LastModified": "2025-01-11T12:00:00Z",
            "ContentLength": len(self.content),
        }

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        self.downloads += 1
        Path(Filename).write_bytes(self.content)
        if self.change_during_download:
            self.revision = "v2"


def test_key_format() -> None:
    assert daily_key(date(2025, 1, 10)) == "us_stocks_sip/day_aggs_v1/2025/01/2025-01-10.csv.gz"


def test_download_cache_and_revisions(tmp_path: Path) -> None:
    content = gzip.compress(b"ticker,open,high,low,close,volume\nAAPL,1,2,1,2,10\n")
    client = FakeClient(content)
    options = dict(
        client=client,
        session=date(2025, 1, 10),
        output_root=tmp_path / "files",
        manifest_path=tmp_path / "manifest.json",
    )
    first = download_daily_flat_file(**options)
    assert first["rows"] == 1
    assert client.downloads == 1
    again = download_daily_flat_file(**options)
    assert again == first
    assert client.downloads == 1
    client.revision = "v2"
    revised = download_daily_flat_file(**options)
    assert revised["sha256"] == first["sha256"]
    assert client.downloads == 2


def test_reject_upstream_revision_during_download(tmp_path: Path) -> None:
    client = FakeClient(gzip.compress(b"a,b\n1,2\n"))
    client.change_during_download = True
    with pytest.raises(SourceChangedError):
        download_daily_flat_file(
            client=client,
            session=date(2025, 1, 10),
            output_root=tmp_path / "files",
            manifest_path=tmp_path / "manifest.json",
        )
    assert not (tmp_path / "manifest.json").exists()
