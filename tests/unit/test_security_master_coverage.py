"""Point-in-time security master coverage regression tests."""

from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import SecurityId
from quant_research.security_master.coverage import audit_security_master
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


def test_coverage_respects_knowledge_cutoff_and_flags_collisions() -> None:
    t = datetime(2025, 1, 1, tzinfo=UTC)
    first, second = SecurityId("SEC-000001"), SecurityId("SEC-000002")
    records = [SecurityRecord(sid, SecurityType.COMMON_STOCK, t) for sid in (first, second)]
    validity = ValidityPeriod(t, t + timedelta(days=30))
    listings = [
        ListingRecord(first, "AAA", "XNAS", validity, t, "test"),
        ListingRecord(second, "AAA", "XNAS", validity, t + timedelta(days=2), "test"),
    ]
    identifiers = [
        IdentifierRecord(first, IdentifierType.SHARE_CLASS_FIGI, "FIGI1", validity, t, "test"),
        IdentifierRecord(
            second,
            IdentifierType.SHARE_CLASS_FIGI,
            "FIGI1",
            validity,
            t + timedelta(days=2),
            "test",
        ),
    ]
    resolver = SecurityResolver(records, identifiers, listings)
    before = audit_security_master(resolver, valid_at=t + timedelta(days=3), known_at=t)
    assert before.active_securities == 1
    assert not before.has_conflicts
    assert before.missing_active_listing == (second,)
    after = audit_security_master(
        resolver, valid_at=t + timedelta(days=3), known_at=t + timedelta(days=3)
    )
    assert after.active_securities == 2
    assert after.has_conflicts
    assert after.ticker_mic_conflicts["AAA:XNAS"] == (first, second)
    assert after.identifier_conflicts["share_class_figi:FIGI1"] == (first, second)


def test_coverage_requires_utc() -> None:
    resolver = SecurityResolver([])
    with pytest.raises(ValueError, match="known_at"):
        audit_security_master(
            resolver,
            valid_at=datetime(2025, 1, 1, tzinfo=UTC),
            known_at=datetime(2025, 1, 1),
        )
