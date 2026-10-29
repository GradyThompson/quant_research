"""Immutable, time-aware security master data contracts.

These records model identifiers and listings, not lookup or entity resolution.
All intervals are half-open: [valid_from, valid_to).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from quant_research.core import SecurityId


class SecurityType(StrEnum):
    COMMON_STOCK = "common_stock"
    PREFERRED_STOCK = "preferred_stock"
    ETF = "etf"
    ADR = "adr"
    OTHER = "other"


class IdentifierType(StrEnum):
    FIGI = "figi"
    SHARE_CLASS_FIGI = "share_class_figi"
    COMPOSITE_FIGI = "composite_figi"
    ISIN = "isin"
    CUSIP = "cusip"
    SEDOL = "sedol"


def _require_utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must have a UTC offset of zero")


@dataclass(frozen=True, slots=True)
class ValidityPeriod:
    """UTC interval [valid_from, valid_to); None means open-ended."""

    valid_from: datetime
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        _require_utc(self.valid_from, "valid_from")
        if self.valid_to is not None:
            _require_utc(self.valid_to, "valid_to")
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be later than valid_from")

    def contains(self, instant: datetime) -> bool:
        _require_utc(instant, "instant")
        return self.valid_from <= instant and (self.valid_to is None or instant < self.valid_to)


@dataclass(frozen=True, slots=True)
class SecurityRecord:
    """Permanent instrument identity; not a company or a ticker."""

    security_id: SecurityId
    security_type: SecurityType
    created_at: datetime
    issuer_id: str | None = None

    def __post_init__(self) -> None:
        _require_utc(self.created_at, "created_at")
        if not isinstance(self.security_id, SecurityId):
            raise ValueError("security_id must be a SecurityId")
        if not isinstance(self.security_type, SecurityType):
            raise ValueError("security_type must be a SecurityType")
        if self.issuer_id is not None and not self.issuer_id.strip():
            raise ValueError("issuer_id cannot be blank")


@dataclass(frozen=True, slots=True)
class IdentifierRecord:
    """An identifier assigned to a security during a validity period.

    recorded_at is when the platform learned this particular assertion.
    Historical corrections must create new records rather than mutate old ones.
    """

    security_id: SecurityId
    identifier_type: IdentifierType
    identifier_value: str
    validity: ValidityPeriod
    recorded_at: datetime
    source: str
    superseded_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.security_id, SecurityId):
            raise ValueError("security_id must be a SecurityId")
        if not isinstance(self.identifier_type, IdentifierType):
            raise ValueError("identifier_type must be an IdentifierType")
        if not self.identifier_value or self.identifier_value != self.identifier_value.strip():
            raise ValueError("identifier_value must be nonblank without surrounding whitespace")
        if not isinstance(self.validity, ValidityPeriod):
            raise ValueError("validity must be a ValidityPeriod")
        _require_utc(self.recorded_at, "recorded_at")
        if self.superseded_at is not None:
            _require_utc(self.superseded_at, "superseded_at")
            if self.superseded_at <= self.recorded_at:
                raise ValueError("superseded_at must be later than recorded_at")
        if not self.source.strip():
            raise ValueError("source cannot be blank")


@dataclass(frozen=True, slots=True)
class ListingRecord:
    """Time-bounded exchange listing and ticker of a security."""

    security_id: SecurityId
    ticker: str
    mic: str
    validity: ValidityPeriod
    recorded_at: datetime
    source: str
    superseded_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.security_id, SecurityId):
            raise ValueError("security_id must be a SecurityId")
        if not self.ticker or self.ticker != self.ticker.strip():
            raise ValueError("ticker must be nonblank without surrounding whitespace")
        if (
            len(self.mic) != 4
            or not self.mic.isascii()
            or not self.mic.isalnum()
            or self.mic != self.mic.upper()
        ):
            raise ValueError("mic must be four uppercase ASCII letters/digits")
        if not isinstance(self.validity, ValidityPeriod):
            raise ValueError("validity must be a ValidityPeriod")
        _require_utc(self.recorded_at, "recorded_at")
        if self.superseded_at is not None:
            _require_utc(self.superseded_at, "superseded_at")
            if self.superseded_at <= self.recorded_at:
                raise ValueError("superseded_at must be later than recorded_at")
        if not self.source.strip():
            raise ValueError("source cannot be blank")
