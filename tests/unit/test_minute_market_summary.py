"""Market-wide research summaries and source-keyed caching."""

import gzip
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.data.ingestion.minute_market_summary import (
    load_or_summarize_market,
    summarize_market_minutes,
)

DAY = date(2024, 6, 10)
OPEN = datetime(2024, 6, 10, 13, 30, tzinfo=UTC)
CLOSE = OPEN + timedelta(hours=6, minutes=30)


def _file(path, *, multiplier=1, duplicate=False):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("ticker,window_start,open,high,low,close,volume\n")
        for i in range(390):
            ns = int((OPEN + timedelta(minutes=i)).timestamp()) * 1_000_000_000
            stream.write(f"AAA,{ns},100,101,99,100,{multiplier * 100}\n")
            if duplicate and i == 1:
                stream.write(f"AAA,{ns},100,101,99,100,100\n")
            if i != 5:
                stream.write(f"BBB,{ns},100,101,99,100,100\n")
    return path


def test_market_summary_excludes_missing_open_minutes(tmp_path):
    source = _file(tmp_path / "daily.gz")
    summaries = summarize_market_minutes(source, opening=OPEN, closing=CLOSE)
    assert list(summaries) == ["AAA"]
    assert summaries["AAA"].opening_volume == 1500
    assert summaries["AAA"].dollar_volume == 3_900_000
    assert summaries["AAA"].last_close == 100


def test_summary_cache_reused_only_for_same_source_sha(tmp_path):
    source = _file(tmp_path / "daily.gz")
    kw = dict(
        cache_root=tmp_path / "cache", day=DAY, source_sha256="a" * 64, opening=OPEN, closing=CLOSE
    )
    first, hit = load_or_summarize_market(source, **kw)
    assert hit is False
    second, hit = load_or_summarize_market(source, **kw)
    assert hit is True and first == second
    other, hit = load_or_summarize_market(source, **{**kw, "source_sha256": "b" * 64})
    assert hit is False and other == first


def test_duplicate_open_minute_is_rejected(tmp_path):
    source = _file(tmp_path / "daily.gz", duplicate=True)
    with pytest.raises(ValueError, match="duplicate opening"):
        summarize_market_minutes(source, opening=OPEN, closing=CLOSE)


def test_eligible_today_not_conditioned_on_future_close(tmp_path):
    source = tmp_path / "halted.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write("ticker,window_start,open,high,low,close,volume\n")
        for i in range(15):
            ns = int((OPEN + timedelta(minutes=i)).timestamp()) * 1_000_000_000
            stream.write(f"HALT,{ns},100,101,99,100,100\n")
    rows = summarize_market_minutes(source, opening=OPEN, closing=CLOSE)
    # A halt occurring later in the session must not influence a 09:46 selector.
    assert "HALT" in rows
    assert not rows["HALT"].close_eligible
