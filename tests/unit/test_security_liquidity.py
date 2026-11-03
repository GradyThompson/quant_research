"""Synthetic daily-bar screening tests."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from quant_research.security_master.liquidity import screen_daily_bars


def _bars(prices: list[float], *, start: date = date(2025, 1, 2)) -> pa.Table:
    ny = ZoneInfo("America/New_York")
    return pa.table(
        {
            "timestamp": [
                datetime.combine(start + timedelta(days=i), datetime.min.time(), tzinfo=ny)
                .replace(hour=16)
                .astimezone(UTC)
                for i in range(len(prices))
            ],
            "close": prices,
            "volume": [2_000_000] * len(prices),
        }
    )


def test_qualifies_using_only_previous_days() -> None:
    sessions = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4))
    result = screen_daily_bars(
        _bars([10.0, 11.0, 12.0, 1.0]),
        evaluation_date=date(2025, 1, 5),
        expected_sessions=sessions,
        min_valid_sessions=3,
    )
    assert result.reason is None
    assert result.last_close == 12
    assert result.mean_dollar_volume == 22_000_000


def test_rejects_missing_latest() -> None:
    sessions = (date(2025, 1, 2), date(2025, 1, 3))
    result = screen_daily_bars(
        _bars([10.0]),
        evaluation_date=date(2025, 1, 4),
        expected_sessions=sessions,
        min_valid_sessions=1,
    )
    assert result.reason == "missing_latest_session"


def test_rejects_future_lookback() -> None:
    with pytest.raises(ValueError, match="future"):
        screen_daily_bars(
            _bars([10]),
            evaluation_date=date(2025, 1, 2),
            expected_sessions=(date(2025, 1, 2),),
            min_valid_sessions=1,
        )
