"""Unit tests for the five-session minute window identifiers."""

from datetime import date

import pytest

from quant_research.data.ingestion.minute_window import previous_sessions, session_dataset_id


def test_previous_sessions_skips_weekend_and_market_closure() -> None:
    assert previous_sessions(end_session=date(2025, 1, 10)) == (
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
        date(2025, 1, 10),
    )


def test_previous_sessions_skips_new_year_holiday() -> None:
    assert previous_sessions(end_session=date(2025, 1, 3), count=3) == (
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    )


def test_previous_sessions_returns_one_session() -> None:
    assert previous_sessions(end_session=date(2025, 1, 10), count=1) == (date(2025, 1, 10),)


def test_previous_sessions_rejects_non_session_and_bad_count() -> None:
    with pytest.raises(ValueError, match="Not a trading session"):
        previous_sessions(end_session=date(2025, 1, 11))
    with pytest.raises(ValueError, match="positive"):
        previous_sessions(end_session=date(2025, 1, 10), count=0)


def test_previous_sessions_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="Insufficient calendar history"):
        previous_sessions(end_session=date(2025, 1, 10), count=100_000)


def test_session_dataset_id_is_valid_and_date_scoped() -> None:
    assert str(session_dataset_id("AAPL", date(2025, 1, 8))) == (
        "market.ohlcv.minute.aapl.s20250108"
    )
    with pytest.raises(ValueError, match="uppercase ASCII"):
        session_dataset_id("Aapl", date(2025, 1, 8))
