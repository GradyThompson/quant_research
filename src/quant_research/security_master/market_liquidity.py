"""Point-in-time-date liquidity screening of pinned market-wide day bars."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date
from typing import cast

import pyarrow as pa

from quant_research.security_master.liquidity import screen_daily_bars


def screen_market_daily_tables(
    *,
    members: list[dict[str, str]],
    daily_tables: Mapping[date, pa.Table],
    evaluation_date: date,
    min_sessions: int = 18,
    min_price: float = 5.0,
    min_dollar_volume: float = 10_000_000.0,
    max_securities: int = 500,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Screen candidate security IDs, never by ticker, over a complete dated window.

    Each date must have a pinned daily dataset; missing *security* observations
    remain missing, rather than being forward-filled. This function never reads
    evaluation-day bars.
    """
    sessions = tuple(sorted(daily_tables))
    if not sessions or sessions[-1] >= evaluation_date:
        raise ValueError("Daily tables must precede evaluation date")
    if min_sessions < 1 or min_sessions > len(sessions) or max_securities < 1:
        raise ValueError("Invalid session or output limit")
    identities = [member["security_id"] for member in members]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate candidate security IDs")
    candidate_ids = set(identities)
    observations: dict[str, list[dict[str, object]]] = defaultdict(list)
    duplicate_ids: set[str] = set()
    for session, table in daily_tables.items():
        required = {"security_id", "session", "timestamp", "close", "volume"}
        if not required.issubset(table.column_names):
            raise ValueError(f"Missing daily fields on {session}")
        seen: set[str] = set()
        for row in table.select(sorted(required)).to_pylist():
            security_id = row["security_id"]
            if security_id not in candidate_ids:
                continue
            if row["session"] != session:
                raise ValueError(f"Incorrect session for {security_id} on {session}")
            if security_id in seen:
                duplicate_ids.add(security_id)
            seen.add(security_id)
            observations[security_id].append(row)
    excluded: Counter[str] = Counter()
    eligible: list[dict[str, object]] = []
    for member in members:
        security_id = member["security_id"]
        if security_id in duplicate_ids:
            excluded["duplicate_sessions"] += 1
            continue
        rows = observations.get(security_id, [])
        if not rows:
            excluded["missing_daily_data"] += 1
            continue
        table = pa.Table.from_pylist(
            [
                {"timestamp": row["timestamp"], "close": row["close"], "volume": row["volume"]}
                for row in rows
            ]
        )
        stats = screen_daily_bars(
            table,
            evaluation_date=evaluation_date,
            expected_sessions=sessions,
            min_price=min_price,
            min_mean_dollar_volume=min_dollar_volume,
            min_valid_sessions=min_sessions,
        )
        if stats.reason is not None:
            excluded[stats.reason] += 1
            continue
        eligible.append(
            {
                **member,
                "reference_price": stats.last_close,
                "mean_dollar_volume": stats.mean_dollar_volume,
                "valid_sessions": stats.valid_sessions,
                "missing_sessions": stats.missing_sessions,
            }
        )
    eligible.sort(
        key=lambda item: (
            -cast(float, item["mean_dollar_volume"]),
            str(item["ticker"]),
            str(item["security_id"]),
        )
    )
    selected = eligible[:max_securities]
    excluded["outside_top_n"] += len(eligible) - len(selected)
    return selected, dict(sorted(excluded.items()))
