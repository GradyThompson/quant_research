"""Offline market-wide liquidity screening regression tests."""

from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.security_master.market_liquidity import screen_market_daily_tables


def _table(day: date, rows: list[tuple[str, float, int]]) -> pa.Table:
    timestamp = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=21)
    return pa.table(
        {
            "security_id": [row[0] for row in rows],
            "session": [day] * len(rows),
            "timestamp": [timestamp] * len(rows),
            "close": [row[1] for row in rows],
            "volume": [row[2] for row in rows],
        }
    )


def test_rank_and_do_not_fill_missing_sessions() -> None:
    dates = (date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8))
    tables = {
        day: _table(
            day,
            [("SEC-1", 10.0, 3_000_000), ("SEC-2", 10.0, 2_000_000)]
            if i != 1
            else [("SEC-1", 10.0, 3_000_000)],
        )
        for i, day in enumerate(dates)
    }
    members = [
        {"security_id": "SEC-1", "ticker": "AAA", "mic": "XNAS"},
        {"security_id": "SEC-2", "ticker": "BBB", "mic": "XNYS"},
    ]
    selected, excluded = screen_market_daily_tables(
        members=members, daily_tables=tables, evaluation_date=date(2025, 1, 9), min_sessions=3
    )
    assert [row["ticker"] for row in selected] == ["AAA"]
    assert excluded["insufficient_history"] == 1


def test_duplicate_security_session_fails_closed() -> None:
    day = date(2025, 1, 8)
    selected, excluded = screen_market_daily_tables(
        members=[{"security_id": "SEC-1", "ticker": "AAA", "mic": "XNAS"}],
        daily_tables={day: _table(day, [("SEC-1", 10, 2_000_000), ("SEC-1", 11, 2_000_000)])},
        evaluation_date=date(2025, 1, 9),
        min_sessions=1,
    )
    assert not selected
    assert excluded["duplicate_sessions"] == 1


def test_reject_future() -> None:
    with pytest.raises(ValueError, match="precede"):
        screen_market_daily_tables(
            members=[],
            daily_tables={date(2025, 1, 9): _table(date(2025, 1, 9), [])},
            evaluation_date=date(2025, 1, 9),
            min_sessions=1,
        )
