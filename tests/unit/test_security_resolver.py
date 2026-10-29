"""Historical lookup rules, including point-in-time corrections."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.security_master import (
    AmbiguousSecurityError,
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityNotFoundError,
    SecurityRecord,
    SecurityResolver,
    SecurityType,
    ValidityPeriod,
)


def t(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=UTC)


S1 = SecurityId("SEC-000001")
S2 = SecurityId("SEC-000002")


def security(sid: SecurityId) -> SecurityRecord:
    return SecurityRecord(sid, SecurityType.COMMON_STOCK, t(2010))


def listing(
    sid: SecurityId,
    symbol: str,
    start: int,
    end: int | None = None,
    *,
    recorded: int = 2010,
    superseded: int | None = None,
) -> ListingRecord:
    return ListingRecord(
        sid,
        symbol,
        "XNYS",
        ValidityPeriod(t(start), t(end) if end else None),
        t(recorded),
        "test",
        t(superseded) if superseded else None,
    )


def test_ticker_change_and_reuse():
    resolver = SecurityResolver(
        [security(S1), security(S2)],
        listings=[
            listing(S1, "OLD", 2010, 2020),
            listing(S1, "NEW", 2020),
            listing(S2, "OLD", 2023),
        ],
    )
    assert (
        resolver.resolve_ticker(ticker="OLD", mic="XNYS", valid_at=t(2019), known_at=t(2025)) == S1
    )
    assert (
        resolver.resolve_ticker(ticker="OLD", mic="XNYS", valid_at=t(2024), known_at=t(2025)) == S2
    )
    assert (
        resolver.resolve_ticker(ticker="NEW", mic="XNYS", valid_at=t(2024), known_at=t(2025)) == S1
    )
    with pytest.raises(SecurityNotFoundError):
        resolver.resolve_ticker(ticker="OLD", mic="XNYS", valid_at=t(2021), known_at=t(2025))


def test_known_at_prevents_backfill_lookahead():
    resolver = SecurityResolver([security(S1)], listings=[listing(S1, "ABC", 2018, recorded=2024)])
    with pytest.raises(SecurityNotFoundError):
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2020), known_at=t(2023))
    assert (
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2020), known_at=t(2025)) == S1
    )


def test_ambiguous_ticker_is_error():
    resolver = SecurityResolver(
        [security(S1), security(S2)], listings=[listing(S1, "ABC", 2010), listing(S2, "ABC", 2010)]
    )
    with pytest.raises(AmbiguousSecurityError):
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2020), known_at=t(2025))


def test_correction_is_known_only_after_supersession():
    resolver = SecurityResolver(
        [security(S1), security(S2)],
        listings=[
            listing(S1, "ABC", 2010, recorded=2010, superseded=2022),
            listing(S2, "ABC", 2010, recorded=2022),
        ],
    )
    assert (
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2019), known_at=t(2021)) == S1
    )
    assert (
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2019), known_at=t(2023)) == S2
    )


def test_identifier_resolution_and_ambiguity():
    identifier = IdentifierRecord(
        S1, IdentifierType.ISIN, "US0000000001", ValidityPeriod(t(2010)), t(2020), "test"
    )
    resolver = SecurityResolver([security(S1)], identifiers=[identifier])
    with pytest.raises(SecurityNotFoundError):
        resolver.resolve_identifier(
            identifier_type=IdentifierType.ISIN,
            identifier_value="US0000000001",
            valid_at=t(2019),
            known_at=t(2019),
        )
    assert (
        resolver.resolve_identifier(
            identifier_type=IdentifierType.ISIN,
            identifier_value="US0000000001",
            valid_at=t(2019),
            known_at=t(2023),
        )
        == S1
    )
    resolver2 = SecurityResolver(
        [security(S1), security(S2)],
        identifiers=[
            identifier,
            IdentifierRecord(
                S2, IdentifierType.ISIN, "US0000000001", ValidityPeriod(t(2010)), t(2020), "test"
            ),
        ],
    )
    with pytest.raises(AmbiguousSecurityError):
        resolver2.resolve_identifier(
            identifier_type=IdentifierType.ISIN,
            identifier_value="US0000000001",
            valid_at=t(2019),
            known_at=t(2023),
        )


def test_duplicate_same_security_not_ambiguous():
    record = listing(S1, "ABC", 2010)
    resolver = SecurityResolver([security(S1)], listings=[record, record])
    assert (
        resolver.resolve_ticker(ticker="ABC", mic="XNYS", valid_at=t(2020), known_at=t(2025)) == S1
    )


def test_unknown_security_rejected():
    with pytest.raises(ValueError, match="Unknown security_id"):
        SecurityResolver([security(S1)], listings=[listing(S2, "ABC", 2010)])


def test_supersession_must_follow_recording():
    with pytest.raises(ValueError, match="superseded_at"):
        listing(S1, "ABC", 2010, recorded=2023, superseded=2022)


def test_invalid_lookup_time_rejected():
    resolver = SecurityResolver([security(S1)])
    with pytest.raises(ValueError, match="UTC"):
        resolver.resolve_ticker(
            ticker="ABC", mic="XNYS", valid_at=datetime(2020, 1, 1), known_at=t(2025)
        )
