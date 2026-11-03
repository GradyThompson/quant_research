"""Deterministic snapshot-derived universe; no market-data or liquidity claims."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from quant_research.core import SecurityId
from quant_research.security_master.models import SecurityType, ValidityPeriod
from quant_research.security_master.resolver import SecurityResolver


@dataclass(frozen=True, slots=True)
class UniverseMember:
    security_id: SecurityId
    ticker: str
    mic: str


@dataclass(frozen=True, slots=True)
class UniverseReport:
    registered: int
    active_listings: int
    common_stock_listings: int
    eligible: int
    excluded_non_common: int
    excluded_ambiguous: int
    excluded_missing_identifier: int


@dataclass(frozen=True, slots=True)
class UniverseResult:
    members: tuple[UniverseMember, ...]
    report: UniverseReport


def build_universe(
    resolver: SecurityResolver,
    *,
    valid_at: datetime,
    known_at: datetime,
    require_share_class_figi: bool = True,
    allowed_mics: frozenset[str] | None = None,
) -> UniverseResult:
    """Select one unambiguous active listing per common stock.

    Historical validity and retrieval-time knowledge are independent filters.
    This is a dated inventory, not a claim of historically available knowledge.
    """
    if valid_at.utcoffset() != timedelta(0):
        raise ValueError("valid_at must be timezone-aware UTC")
    if known_at.utcoffset() != timedelta(0):
        raise ValueError("known_at must be timezone-aware UTC")

    def is_visible(
        validity: ValidityPeriod, recorded_at: datetime, superseded_at: datetime | None
    ) -> bool:
        return (
            validity.contains(valid_at)
            and recorded_at <= known_at
            and (superseded_at is None or known_at < superseded_at)
        )

    type_by_id = {security.security_id: security.security_type for security in resolver.securities}
    valid_identifiers = {
        record.security_id
        for record in resolver.identifiers
        if record.identifier_type.value == "share_class_figi"
        and is_visible(record.validity, record.recorded_at, record.superseded_at)
    }
    active = [
        listing
        for listing in resolver.listings
        if is_visible(listing.validity, listing.recorded_at, listing.superseded_at)
        and (allowed_mics is None or listing.mic in allowed_mics)
    ]
    common = [
        listing
        for listing in active
        if type_by_id[listing.security_id] is SecurityType.COMMON_STOCK
    ]
    noncommon = len(active) - len(common)
    missing = [
        listing
        for listing in common
        if require_share_class_figi and listing.security_id not in valid_identifiers
    ]
    candidates = [
        listing
        for listing in common
        if not require_share_class_figi or listing.security_id in valid_identifiers
    ]
    by_security: dict[SecurityId, set[tuple[str, str]]] = {}
    by_listing: dict[tuple[str, str], set[SecurityId]] = {}
    for listing in candidates:
        by_security.setdefault(listing.security_id, set()).add((listing.ticker, listing.mic))
        by_listing.setdefault((listing.ticker, listing.mic), set()).add(listing.security_id)
    members: list[UniverseMember] = []
    ambiguous_ids: set[SecurityId] = set()
    for listing in candidates:
        if (
            len(by_security[listing.security_id]) != 1
            or len(by_listing[(listing.ticker, listing.mic)]) != 1
        ):
            ambiguous_ids.add(listing.security_id)
            continue
        members.append(UniverseMember(listing.security_id, listing.ticker, listing.mic))
    unique = {(m.security_id, m.ticker, m.mic): m for m in members}
    ordered = tuple(sorted(unique.values(), key=lambda m: (m.ticker, m.mic, str(m.security_id))))
    return UniverseResult(
        ordered,
        UniverseReport(
            registered=len(resolver.securities),
            active_listings=len(active),
            common_stock_listings=len(common),
            eligible=len(ordered),
            excluded_non_common=noncommon,
            excluded_ambiguous=len(ambiguous_ids),
            excluded_missing_identifier=len({x.security_id for x in missing}),
        ),
    )
