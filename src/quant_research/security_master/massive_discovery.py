"""Discover a dated Massive common-stock inventory without inferred continuity.

Historical provider snapshots are recorded *now*. They are not evidence that
this research system knew the identifiers on the historical validity date.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from quant_research.security_master.massive_bootstrap import _get, _security_id
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


class TickerInventoryClient(Protocol):
    def list_tickers(self, **kwargs: Any) -> Iterable[Any]: ...


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    as_of: date
    returned: int
    included: int
    missing_identity: int
    missing_exchange: int
    other_locale_or_type: int
    duplicate_rows: int
    conflicting_figis: int
    conflicting_listings: int
    excluded_conflict_rows: int


def discover_massive_common_stocks(
    client: TickerInventoryClient,
    *,
    as_of: date,
    recorded_at: datetime,
    active: bool | None = None,
) -> tuple[SecurityResolver, DiscoveryReport]:
    """Create a one-day US common-stock inventory from all provider pages.

    `active=None` queries both active and inactive flags. Missing FIGIs are
    reported and skipped; conflicting ticker or FIGI identities are quarantined.
    Returned records assert validity only on `as_of`, not across other dates.
    """
    if recorded_at.tzinfo is None or recorded_at.utcoffset() != timedelta(0):
        raise ValueError("recorded_at must be timezone-aware UTC")
    if as_of > recorded_at.date():
        raise ValueError("Cannot discover future date")
    start = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
    period = ValidityPeriod(start, start + timedelta(days=1))
    valid_rows: list[tuple[str, str, str]] = []
    returned = missing_identity = missing_exchange = other = duplicates = 0
    statuses = (True, False) if active is None else (active,)
    for status in statuses:
        # Massive's SDK iterates pagination; do not cap to its first page.
        for row in client.list_tickers(
            market="stocks",
            type="CS",
            date=as_of.isoformat(),
            active=status,
            limit=1000,
        ):
            returned += 1
            if _get(row, "locale") != "us" or _get(row, "type") != "CS":
                other += 1
                continue
            figi = _get(row, "share_class_figi")
            if not isinstance(figi, str) or not figi.strip():
                missing_identity += 1
                continue
            ticker = _get(row, "ticker")
            mic = _get(row, "primary_exchange")
            if (
                not isinstance(ticker, str)
                or not ticker
                or ticker != ticker.strip()
                or not isinstance(mic, str)
                or len(mic) != 4
                or not mic.isascii()
                or not mic.isalnum()
                or mic != mic.upper()
            ):
                missing_exchange += 1
                continue
            valid_rows.append((figi, ticker, mic))

    # A FIGI can be associated with multiple ticker/MIC pairs in a dated
    # provider response. Without effective-time evidence we cannot choose
    # which assertion is correct. Quarantine *all* implicated identities.
    from collections import defaultdict

    listings_by_figi: dict[str, set[tuple[str, str]]] = defaultdict(set)
    figis_by_listing: dict[tuple[str, str], set[str]] = defaultdict(set)
    for figi, ticker, mic in valid_rows:
        listings_by_figi[figi].add((ticker, mic))
        figis_by_listing[(ticker, mic)].add(figi)
    conflicted_figis = {figi for figi, keys in listings_by_figi.items() if len(keys) > 1}
    conflicted_keys = {key for key, figis in figis_by_listing.items() if len(figis) > 1}
    # Quarantine whole identity/listing components so no surviving assertion
    # can depend on an identity implicated in a conflict.
    while True:
        new_figis = conflicted_figis | {
            figi for key in conflicted_keys for figi in figis_by_listing[key]
        }
        new_keys = conflicted_keys | {key for figi in new_figis for key in listings_by_figi[figi]}
        if new_figis == conflicted_figis and new_keys == conflicted_keys:
            break
        conflicted_figis, conflicted_keys = new_figis, new_keys

    securities: dict[str, SecurityRecord] = {}
    listings: list[ListingRecord] = []
    identifiers: list[IdentifierRecord] = []
    excluded_conflict_rows = 0
    for figi, ticker, mic in valid_rows:
        if figi in conflicted_figis or (ticker, mic) in conflicted_keys:
            excluded_conflict_rows += 1
            continue
        if figi in securities:
            duplicates += 1
            continue
        sid = _security_id(figi)
        securities[figi] = SecurityRecord(sid, SecurityType.COMMON_STOCK, recorded_at)
        source = "massive.reference.tickers.dated_inventory"
        identifiers.append(
            IdentifierRecord(
                sid,
                IdentifierType.SHARE_CLASS_FIGI,
                figi,
                period,
                recorded_at,
                source,
            )
        )
        listings.append(ListingRecord(sid, ticker, mic, period, recorded_at, source))
    if not securities:
        raise ValueError("No identifiable common stocks found; refusing empty snapshot")
    return SecurityResolver(securities.values(), identifiers, listings), DiscoveryReport(
        as_of,
        returned,
        len(securities),
        missing_identity,
        missing_exchange,
        other,
        duplicates,
        len(conflicted_figis),
        len(conflicted_keys),
        excluded_conflict_rows,
    )
