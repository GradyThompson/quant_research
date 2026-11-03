"""Optimized research parser parity, cache safety, and invalidation."""

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.data.ingestion.minute_research_extract import (
    load_or_parse_selected_minutes,
    parse_selected_minutes_fast,
)

DAY = date(2024, 10, 15)
OPENING = datetime(2024, 10, 15, 13, 30, tzinfo=UTC)
CLOSING = OPENING + timedelta(hours=6, minutes=30)
INGESTED = datetime(2026, 1, 1, tzinfo=UTC)
NS = int(OPENING.timestamp()) * 1_000_000_000


def _file(tmp_path, header, values):
    path = tmp_path / "market.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(header + "\n")
        for value in values:
            handle.write(value + "\n")
    return path


def _kwargs():
    return {
        "session": DAY,
        "tickers": {"AAPL", "QQQ"},
        "opening": OPENING,
        "closing": CLOSING,
        "publication_lag_seconds": 60,
        "ingested_at": INGESTED,
    }


def test_fast_parser_filters_market_and_preserves_publication_timing(tmp_path):
    path = _file(
        tmp_path,
        "ticker,window_start,open,high,low,close,volume",
        [
            f"OTHER,{NS},10,11,9,10.5,500",
            f"AAPL,{NS},10,11,9,10.5,100",
            f'"QQQ",{NS},20,21,19,20.5,200',
            f"AAPL,{NS + 60_000_000_000},11,12,10,11.5,101",
        ],
    )
    records = parse_selected_minutes_fast(path, **_kwargs())
    assert len(records["AAPL"]) == 2
    assert len(records["QQQ"]) == 1
    assert records["AAPL"][0]["close"] == 10.5
    assert records["AAPL"][0]["timestamp"] == OPENING + timedelta(minutes=1)
    assert records["AAPL"][0]["available_at"] == OPENING + timedelta(minutes=2)
    assert records["AAPL"][1]["timestamp"] > records["AAPL"][0]["timestamp"]


def test_fast_parser_accepts_arbitrary_header_order(tmp_path):
    path = _file(
        tmp_path,
        "window_start,open,high,low,close,volume,ticker",
        [f"{NS},10,11,9,10.5,100,AAPL", f"{NS},10,11,9,10.5,200,OTHER"],
    )
    records = parse_selected_minutes_fast(path, **_kwargs())
    assert len(records["AAPL"]) == 1
    assert records["QQQ"] == []


def test_fast_parser_rejects_selected_duplicate_and_invalid_prices(tmp_path):
    row = f"AAPL,{NS},10,11,9,10.5,100"
    duplicate = _file(tmp_path, "ticker,window_start,open,high,low,close,volume", [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_selected_minutes_fast(duplicate, **_kwargs())
    invalid = _file(
        tmp_path,
        "ticker,window_start,open,high,low,close,volume",
        [f"AAPL,{NS},10,9,8,10.5,100"],
    )
    with pytest.raises(ValueError, match="Invalid OHLCV"):
        parse_selected_minutes_fast(invalid, **_kwargs())


def test_cache_reuses_rows_without_parser_and_invalidates_source_revision(tmp_path):
    path = _file(
        tmp_path,
        "ticker,window_start,open,high,low,close,volume",
        [f"AAPL,{NS},10,11,9,10.5,100"],
    )
    cache_root = tmp_path / "cache"
    calls: list[int] = []

    def counting_parser(*args, **kwargs):
        calls.append(1)
        return parse_selected_minutes_fast(*args, **kwargs)

    params = {
        "cache_root": cache_root,
        "source_sha256": "a" * 64,
        "parser": counting_parser,
        **_kwargs(),
    }
    first, hit = load_or_parse_selected_minutes(path, **params)
    assert not hit
    second, hit = load_or_parse_selected_minutes(path, **params)
    assert hit
    assert first == second
    assert len(calls) == 1
    third, hit = load_or_parse_selected_minutes(path, **{**params, "source_sha256": "b" * 64})
    assert not hit
    assert third == first
    assert len(calls) == 2
    _, hit = load_or_parse_selected_minutes(path, **{**params, "publication_lag_seconds": 120})
    assert not hit
    assert len(calls) == 3


def test_cache_corruption_fails_closed(tmp_path):
    path = _file(
        tmp_path,
        "ticker,window_start,open,high,low,close,volume",
        [f"AAPL,{NS},10,11,9,10.5,100"],
    )
    params = {"cache_root": tmp_path / "cache", "source_sha256": "a" * 64, **_kwargs()}
    load_or_parse_selected_minutes(path, **params)
    (cache_file,) = (tmp_path / "cache").rglob("*.json.gz")
    cache_file.write_bytes(b"corrupt gzip")
    with pytest.raises((OSError, EOFError)):
        load_or_parse_selected_minutes(path, **params)
