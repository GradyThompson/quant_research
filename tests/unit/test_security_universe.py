"""Point-in-time inventory eligibility and ambiguity checks."""

from datetime import UTC, datetime, timedelta

from quant_research.core import SecurityId
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver
from quant_research.security_master.universe import build_universe


def test_universe_filters_as_of_and_rejects_ambiguous_ids() -> None:
    day = datetime(2025, 1, 13, 12, tzinfo=UTC)
    known = day + timedelta(days=10)
    span = ValidityPeriod(day - timedelta(hours=12), day + timedelta(hours=12))
    ids = [SecurityId(f"SEC-{i:06d}") for i in range(1, 5)]
    securities = [SecurityRecord(sid, SecurityType.COMMON_STOCK, day) for sid in ids]
    listings = [
        ListingRecord(ids[0], "AAA", "XNAS", span, known, "test"),
        ListingRecord(ids[1], "BBB", "XNYS", span, known, "test"),
        ListingRecord(ids[2], "CCC", "XNAS", span, known, "test"),
        ListingRecord(ids[3], "CCC", "XNAS", span, known, "test"),
    ]
    identifiers = [
        IdentifierRecord(sid, IdentifierType.SHARE_CLASS_FIGI, f"BBG{i:09d}", span, known, "test")
        for i, sid in enumerate(ids)
    ]
    resolver = SecurityResolver(securities, identifiers, listings)
    invisible = build_universe(resolver, valid_at=day, known_at=day)
    assert invisible.report.eligible == 0
    result = build_universe(resolver, valid_at=day, known_at=known)
    assert [(m.ticker, m.mic) for m in result.members] == [("AAA", "XNAS"), ("BBB", "XNYS")]
    assert result.report.excluded_ambiguous == 2
    narrowed = build_universe(
        resolver, valid_at=day, known_at=known, allowed_mics=frozenset({"XNYS"})
    )
    assert [m.ticker for m in narrowed.members] == ["BBB"]
