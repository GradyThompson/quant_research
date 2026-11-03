"""Offline market-wide minute source parsing tests."""

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.data.ingestion.minute_bulk_import import parse_selected_minutes
from quant_research.data.ingestion.minute_flat_files import minute_key


def _file(tmp_path, rows):
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("ticker,window_start,open,high,low,close,volume\n")
        for row in rows:
            file.write(row + "\n")
    return path


def test_minute_key():
    assert (
        minute_key(date(2024, 10, 15)) == "us_stocks_sip/minute_aggs_v1/2024/10/2024-10-15.csv.gz"
    )


def test_selected_minute_end_timestamp_and_filter(tmp_path):
    start = datetime(2024, 10, 15, 13, 30, tzinfo=UTC)
    ns = int(start.timestamp()) * 1_000_000_000
    source = _file(tmp_path, [f"AAPL,{ns},10,11,9,10.5,100", f"OTHER,{ns},10,11,9,10.5,200"])
    rows = parse_selected_minutes(
        source,
        session=date(2024, 10, 15),
        tickers={"AAPL"},
        opening=start,
        closing=start + timedelta(hours=6, minutes=30),
        publication_lag_seconds=60,
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(rows["AAPL"]) == 1
    assert rows["AAPL"][0]["timestamp"] == start + timedelta(minutes=1)
    assert rows["AAPL"][0]["available_at"] == start + timedelta(minutes=2)


def test_duplicate_minute_fails(tmp_path):
    start = datetime(2024, 10, 15, 13, 30, tzinfo=UTC)
    ns = int(start.timestamp()) * 1_000_000_000
    source = _file(tmp_path, [f"AAPL,{ns},10,11,9,10.5,100"] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_selected_minutes(
            source,
            session=date(2024, 10, 15),
            tickers={"AAPL"},
            opening=start,
            closing=start + timedelta(hours=6, minutes=30),
            publication_lag_seconds=60,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
