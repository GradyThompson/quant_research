"""Build historical minute-pilot resolution from dated, pinned snapshots."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion, SecurityId
from quant_research.security_master.models import IdentifierRecord, ListingRecord, SecurityRecord
from quant_research.security_master.persistence import SecurityMasterRepository
from quant_research.security_master.resolver import (
    SecurityNotFoundError,
    SecurityResolver,
)


def resolve_historical_member(
    resolver: SecurityResolver,
    member: dict[str, Any],
    *,
    valid_at: datetime,
    known_at: datetime,
) -> SecurityId:
    """Verify identity using the historical listing MIC, not the current MIC.

    Only exact ticker/MIC listings in the pinned snapshot are considered.
    A listing exchange change is permitted; a security identity change or
    multiple active ticker/MIC matches is not.
    """
    ticker = str(member["ticker"])
    expected_id = str(member["security_id"])
    mics = {record.mic for record in resolver.listings if record.ticker == ticker}
    matches: set[SecurityId] = set()
    for mic in sorted(mics):
        try:
            matches.add(
                resolver.resolve_ticker(
                    ticker=ticker,
                    mic=mic,
                    valid_at=valid_at,
                    known_at=known_at,
                )
            )
        except SecurityNotFoundError:
            continue
    if len(matches) != 1:
        raise ValueError(
            f"Expected one active historical identity for {ticker} at "
            f"{valid_at.isoformat()}, found {sorted(map(str, matches))}"
        )
    actual = next(iter(matches))
    if str(actual) != expected_id:
        raise ValueError(
            f"Reference identity changed for {ticker} at {valid_at.isoformat()}: "
            f"expected {expected_id}, got {actual}"
        )
    return actual


def build_minute_pilot_resolver(
    plan: dict[str, Any], repository: SecurityMasterRepository
) -> SecurityResolver:
    """Combine dated listings after verifying every planned security-session.

    The evaluation snapshot covers only the evaluation date. Initialization
    sessions require their own pinned daily reference snapshots.
    """
    evaluation = date.fromisoformat(plan["evaluation_session"])
    sessions = [date.fromisoformat(value) for value in plan["initialization_sessions"]]
    sessions.append(evaluation)
    snapshots: dict[date, str] = {}
    for item in plan["source_daily_datasets"]:
        day = date.fromisoformat(item["session"])
        if day in snapshots:
            raise ValueError(f"Duplicate daily snapshot reference: {day}")
        snapshots[day] = item["snapshot_version"]
    snapshots[evaluation] = plan["security_master_snapshot_version"]
    missing = [day for day in sessions if day not in snapshots]
    if missing:
        raise ValueError(f"No pinned reference snapshot for pilot sessions: {missing}")

    securities: dict[SecurityId, SecurityRecord] = {}
    listings: list[ListingRecord] = []
    identifiers: list[IdentifierRecord] = []
    for day in sessions:
        resolver = repository.load(DatasetVersion(UUID(snapshots[day])))
        known_at = datetime.now(UTC)
        opening = xcals.get_calendar("XNYS").session_open(pd.Timestamp(day)).to_pydatetime()
        for member in plan["members"]:
            resolve_historical_member(resolver, member, valid_at=opening, known_at=known_at)
        for record in resolver.securities:
            previous = securities.get(record.security_id)
            if previous is not None and previous.security_type != record.security_type:
                raise ValueError(f"Conflicting security type: {record.security_id}")
            securities.setdefault(record.security_id, record)
        listings.extend(resolver.listings)
        identifiers.extend(resolver.identifiers)
    return SecurityResolver(securities.values(), identifiers, listings)
