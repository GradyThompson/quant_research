"""Point-in-time coverage and ambiguity audit for a security-master snapshot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from quant_research.core import SecurityId
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
)
from quant_research.security_master.resolver import SecurityResolver


@dataclass(frozen=True, slots=True)
class CoverageReport:
    valid_at: datetime
    known_at: datetime
    registered_securities: int
    active_securities: int
    active_listings: int
    active_identifiers: int
    active_by_type: dict[str, int]
    missing_share_class_figi: tuple[SecurityId, ...]
    missing_active_listing: tuple[SecurityId, ...]
    ticker_mic_conflicts: dict[str, tuple[SecurityId, ...]]
    identifier_conflicts: dict[str, tuple[SecurityId, ...]]
    multi_listing_securities: tuple[SecurityId, ...]

    @property
    def has_conflicts(self) -> bool:
        return bool(self.ticker_mic_conflicts or self.identifier_conflicts)


def audit_security_master(
    resolver: SecurityResolver, *, valid_at: datetime, known_at: datetime
) -> CoverageReport:
    """Audit *known* assertions valid at a historical instant.

    This does not infer missing securities from an incomplete master. A known-at
    cutoff after the test date is a retrospective audit, not a PIT simulation.
    """
    for label, instant in (("valid_at", valid_at), ("known_at", known_at)):
        if (
            not isinstance(instant, datetime)
            or instant.tzinfo is None
            or instant.utcoffset() != UTC.utcoffset(instant)
        ):
            raise ValueError(f"{label} must be UTC-aware")

    def visible(record: object) -> bool:
        # Both supported assertion classes expose these validated properties.
        if not isinstance(record, (IdentifierRecord, ListingRecord)):
            raise TypeError("Unsupported assertion")
        return (
            record.validity.contains(valid_at)
            and record.recorded_at <= known_at
            and (record.superseded_at is None or known_at < record.superseded_at)
        )

    listings = [row for row in resolver.listings if visible(row)]
    identifiers = [row for row in resolver.identifiers if visible(row)]
    listings_by_id: dict[SecurityId, int] = defaultdict(int)
    ticker_map: dict[tuple[str, str], set[SecurityId]] = defaultdict(set)
    identifier_map: dict[tuple[IdentifierType, str], set[SecurityId]] = defaultdict(set)
    for listing in listings:
        listings_by_id[listing.security_id] += 1
        ticker_map[(listing.ticker, listing.mic)].add(listing.security_id)
    for identifier in identifiers:
        identifier_map[(identifier.identifier_type, identifier.identifier_value)].add(
            identifier.security_id
        )
    active_ids = set(listings_by_id)
    types = {row.security_id: row.security_type for row in resolver.securities}
    counts: dict[str, int] = defaultdict(int)
    for sid in active_ids:
        counts[types[sid].value] += 1
    known_figis = {
        row.security_id
        for row in identifiers
        if row.identifier_type == IdentifierType.SHARE_CLASS_FIGI
    }
    return CoverageReport(
        valid_at=valid_at,
        known_at=known_at,
        registered_securities=len(resolver.securities),
        active_securities=len(active_ids),
        active_listings=len(listings),
        active_identifiers=len(identifiers),
        active_by_type=dict(sorted(counts.items())),
        missing_share_class_figi=tuple(sorted(active_ids - known_figis, key=str)),
        missing_active_listing=tuple(sorted((set(types) - active_ids), key=str)),
        ticker_mic_conflicts={
            f"{ticker}:{mic}": tuple(sorted(ids, key=str))
            for (ticker, mic), ids in sorted(ticker_map.items())
            if len(ids) > 1
        },
        identifier_conflicts={
            f"{kind.value}:{value}": tuple(sorted(ids, key=str))
            for (kind, value), ids in sorted(identifier_map.items())
            if len(ids) > 1
        },
        multi_listing_securities=tuple(
            sorted((sid for sid, count in listings_by_id.items() if count > 1), key=str)
        ),
    )
