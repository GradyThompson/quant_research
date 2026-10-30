"""Trading-session selection and dataset identities for minute pilot windows."""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetId


def previous_sessions(*, end_session: date, count: int = 5) -> tuple[date, ...]:
    """Return ``count`` exchange sessions ending on ``end_session`` (inclusive)."""
    if count < 1:
        raise ValueError("count must be positive")

    calendar = xcals.get_calendar("XNYS")
    end = pd.Timestamp(end_session)
    if not calendar.is_session(end):
        raise ValueError(f"Not a trading session: {end_session}")

    end_index = calendar.sessions.get_loc(end)
    if not isinstance(end_index, int):
        raise RuntimeError("Expected a unique exchange session")
    if end_index + 1 < count:
        raise ValueError("Insufficient calendar history")

    selected = calendar.sessions[end_index + 1 - count : end_index + 1]
    return tuple(session.date() for session in selected)


def session_dataset_id(ticker: str, session: date) -> DatasetId:
    """Give every session its own immutable dataset identity."""
    if not ticker.isascii() or not ticker.isalpha() or not ticker.isupper():
        raise ValueError("ticker must be uppercase ASCII letters")
    return DatasetId(f"market.ohlcv.minute.{ticker.lower()}.s{session:%Y%m%d}")
