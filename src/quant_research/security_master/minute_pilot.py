"""Deterministic minute-ingestion pilot from a pinned liquidity screen."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import TypedDict


class PilotMember(TypedDict):
    security_id: str
    ticker: str
    mic: str
    mean_dollar_volume: float


def select_minute_pilot(
    members: list[dict[str, object]], *, count: int = 20
) -> tuple[list[PilotMember], dict[str, int]]:
    """Select highest-liquidity identities supported by the minute REST adapter.

    Input ranking is verified rather than trusted. No fallback to unsupported
    instruments or substitution of ticker for security identity is permitted.
    """
    if count < 1:
        raise ValueError("count must be positive")
    accepted: list[PilotMember] = []
    excluded = {"unsupported_mic": 0, "unsupported_ticker": 0}
    seen_ids: set[str] = set()
    seen_symbols: set[tuple[str, str]] = set()
    for member in members:
        security_id = member.get("security_id")
        ticker = member.get("ticker")
        mic = member.get("mic")
        volume = member.get("mean_dollar_volume")
        if not isinstance(security_id, str) or not security_id:
            raise ValueError("Invalid security_id in screened universe")
        if not isinstance(ticker, str) or not isinstance(mic, str):
            raise ValueError("Invalid ticker or MIC in screened universe")
        if security_id in seen_ids or (ticker, mic) in seen_symbols:
            raise ValueError("Duplicate security or listing in screened universe")
        seen_ids.add(security_id)
        seen_symbols.add((ticker, mic))
        if not isinstance(volume, (int, float)) or isinstance(volume, bool):
            raise ValueError("Invalid mean_dollar_volume")
        if not isfinite(volume) or volume < 0:
            raise ValueError("Invalid mean_dollar_volume")
        if mic not in ("XNAS", "XNYS"):
            excluded["unsupported_mic"] += 1
            continue
        if not ticker.isascii() or not ticker.isalpha() or not ticker.isupper():
            excluded["unsupported_ticker"] += 1
            continue
        accepted.append(
            {
                "security_id": security_id,
                "ticker": ticker,
                "mic": mic,
                "mean_dollar_volume": float(volume),
            }
        )
    accepted.sort(key=lambda m: (-m["mean_dollar_volume"], m["ticker"], m["mic"], m["security_id"]))
    if len(accepted) < count:
        raise ValueError(f"Only {len(accepted)} supported securities; requested {count}")
    return accepted[:count], excluded


def minute_session_plan(
    *, evaluation_date: date, initialization_sessions: int = 5
) -> tuple[date, ...]:
    """Exchange sessions, including evaluation, strictly excluding earlier future data."""
    if initialization_sessions < 1:
        raise ValueError("initialization_sessions must be positive")
    import exchange_calendars as xcals
    import pandas as pd

    calendar = xcals.get_calendar("XNYS")
    target = pd.Timestamp(evaluation_date)
    if not calendar.is_session(target):
        raise ValueError("Evaluation date is not an exchange session")
    index = calendar.sessions.get_loc(target)
    if not isinstance(index, int) or index < initialization_sessions:
        raise ValueError("Insufficient calendar history")
    return tuple(
        item.date() for item in calendar.sessions[index - initialization_sessions : index + 1]
    )
