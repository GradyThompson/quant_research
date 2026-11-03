"""Tests for historical market-wide daily import and identity quarantine."""

from datetime import UTC, date, datetime, timedelta
from gzip import open as gzip_open
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_research.core import SecurityId
from quant_research.data.ingestion.massive_daily_import import parse_daily_flat_file
from quant_research.security_master.models import (
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


def test_maps_unique_ticker_and_quarantines_unmapped_or_ambiguous(tmp_path: Path) -> None:
    session = date(2025, 1, 10)
    start = datetime(2025, 1, 10, tzinfo=ZoneInfo("America/New_York"))
    validity = ValidityPeriod(start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC))
    created = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [SecurityId(f"SEC-{i:06d}") for i in range(3)]
    resolver = SecurityResolver(
        [SecurityRecord(sid, SecurityType.COMMON_STOCK, created, None) for sid in ids],
        listings=[
            ListingRecord(ids[0], "AAA", "XNAS", validity, created, "test", None),
            ListingRecord(ids[1], "DUP", "XNAS", validity, created, "test", None),
            ListingRecord(ids[2], "DUP", "XNYS", validity, created, "test", None),
        ],
    )
    source = tmp_path / "day.csv.gz"
    with gzip_open(source, "wt") as f:
        f.write("ticker,open,high,low,close,volume,window_start\n")
        for ticker in ("AAA", "DUP", "NONE"):
            f.write(f"{ticker},10,12,9,11,200,{int(start.timestamp()) * 1_000_000_000}\n")
    table, audit = parse_daily_flat_file(source=source, session=session, resolver=resolver)
    assert table.num_rows == 1
    assert table.column("timestamp").to_pylist() == [start.astimezone(UTC)]
    assert table.column("ticker").to_pylist() == ["AAA"]
    assert audit.source_rows == 3
    assert audit.mapped_rows == 1
    assert audit.ambiguous_tickers == 1
    assert audit.unmapped_tickers == 1


def test_rejects_unrecognized_headers(tmp_path: Path) -> None:
    source = tmp_path / "wrong.csv.gz"
    with gzip_open(source, "wt") as f:
        f.write("ticker,close\nAAA,12\n")
    with pytest.raises(ValueError, match="Missing Massive"):
        parse_daily_flat_file(
            source=source,
            session=date(2025, 1, 10),
            resolver=SecurityResolver([]),
        )


def test_rejects_wrong_market_session_timestamp(tmp_path: Path) -> None:
    session = date(2025, 1, 10)
    created = datetime(2026, 1, 1, tzinfo=UTC)
    opening = datetime(2025, 1, 10, 1, tzinfo=ZoneInfo("America/New_York"))
    sid = SecurityId("SEC-000001")
    resolver = SecurityResolver(
        [SecurityRecord(sid, SecurityType.COMMON_STOCK, created, None)],
        listings=[
            ListingRecord(
                sid,
                "AAA",
                "XNAS",
                ValidityPeriod(
                    datetime(2025, 1, 10, tzinfo=UTC), datetime(2025, 1, 11, tzinfo=UTC)
                ),
                created,
                "test",
                None,
            )
        ],
    )
    source = tmp_path / "bad_time.csv.gz"
    with gzip_open(source, "wt") as f:
        f.write("ticker,open,high,low,close,volume,window_start\n")
        f.write(f"AAA,10,12,9,11,200,{int(opening.timestamp()) * 1_000_000_000}\n")
    with pytest.raises(ValueError, match="No mapped bars"):
        parse_daily_flat_file(source=source, session=session, resolver=resolver)
