"""Conservative historical lookup over immutable reference-data assertions.

`valid_at` is when the assertion is effective. `known_at` is the historical
knowledge cutoff. Conflicting active assertions are *never* auto-merged.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from quant_research.core import SecurityId
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
)


class SecurityResolutionError(ValueError):
    """Base class for invalid or unresolvable security lookups."""


class SecurityNotFoundError(SecurityResolutionError):
    """No matching known assertion at the requested times."""


class AmbiguousSecurityError(SecurityResolutionError):
    """Several distinct securities match the requested historical key."""


def _require_utc(instant: datetime, name: str) -> None:
    if (
        not isinstance(instant, datetime)
        or instant.tzinfo is None
        or instant.utcoffset() != UTC.utcoffset(instant)
    ):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _visible(
    record: IdentifierRecord | ListingRecord, *, valid_at: datetime, known_at: datetime
) -> bool:
    return (
        record.validity.contains(valid_at)
        and record.recorded_at <= known_at
        and (record.superseded_at is None or known_at < record.superseded_at)
    )


@dataclass(frozen=True, slots=True)
class SecurityResolver:
    """Immutable in-memory resolver; no fuzzy matches or provider priority.

    All security IDs referenced by assertions must be registered. This is a
    local prototype; persistent versioned security master comes next.
    """

    securities: tuple[SecurityRecord, ...]
    identifiers: tuple[IdentifierRecord, ...]
    listings: tuple[ListingRecord, ...]

    def __init__(
        self,
        securities: Iterable[SecurityRecord],
        identifiers: Iterable[IdentifierRecord] = (),
        listings: Iterable[ListingRecord] = (),
    ) -> None:
        security_tuple = tuple(securities)
        identifier_tuple = tuple(identifiers)
        listing_tuple = tuple(listings)
        known_ids = [s.security_id for s in security_tuple]
        if len(set(known_ids)) != len(known_ids):
            raise ValueError("Duplicate SecurityRecord security_id")
        for identifier in identifier_tuple:
            if identifier.security_id not in known_ids:
                raise ValueError(f"Unknown security_id: {identifier.security_id}")

        for listing in listing_tuple:
            if listing.security_id not in known_ids:
                raise ValueError(f"Unknown security_id: {listing.security_id}")
        object.__setattr__(self, "securities", security_tuple)
        object.__setattr__(self, "identifiers", identifier_tuple)
        object.__setattr__(self, "listings", listing_tuple)

    @staticmethod
    def _unique(matches: set[SecurityId], label: str) -> SecurityId:
        if not matches:
            raise SecurityNotFoundError(f"No security found for {label}")
        if len(matches) > 1:
            raise AmbiguousSecurityError(
                f"Ambiguous security for {label}: {sorted(map(str, matches))}"
            )
        return next(iter(matches))

    def resolve_ticker(
        self, *, ticker: str, mic: str, valid_at: datetime, known_at: datetime
    ) -> SecurityId:
        """Resolve an exact ticker + MIC, never ticker-only."""
        _require_utc(valid_at, "valid_at")
        _require_utc(known_at, "known_at")
        if not ticker or ticker != ticker.strip() or not mic or mic != mic.strip():
            raise ValueError("ticker and mic must be nonblank and trimmed")
        matching = {
            r.security_id
            for r in self.listings
            if r.ticker == ticker
            and r.mic == mic
            and _visible(r, valid_at=valid_at, known_at=known_at)
        }
        return self._unique(matching, f"ticker={ticker!r} mic={mic!r}")

    def resolve_identifier(
        self,
        *,
        identifier_type: IdentifierType,
        identifier_value: str,
        valid_at: datetime,
        known_at: datetime,
    ) -> SecurityId:
        """Resolve a typed external identifier with exact matching."""
        _require_utc(valid_at, "valid_at")
        _require_utc(known_at, "known_at")
        if not isinstance(identifier_type, IdentifierType):
            raise ValueError("identifier_type must be IdentifierType")
        if not identifier_value or identifier_value != identifier_value.strip():
            raise ValueError("identifier_value must be nonblank and trimmed")
        matching = {
            r.security_id
            for r in self.identifiers
            if r.identifier_type == identifier_type
            and r.identifier_value == identifier_value
            and _visible(r, valid_at=valid_at, known_at=known_at)
        }
        return self._unique(matching, f"{identifier_type.value}={identifier_value!r}")
