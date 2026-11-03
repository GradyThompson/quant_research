"""Dated reference resolver regression tests; no provider or store required."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from quant_research.core import SecurityId
from quant_research.data.ingestion.minute_pilot_resolver import (
    build_minute_pilot_resolver,
    resolve_historical_member,
)
from quant_research.security_master.models import (
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


class FakeRepository:
    def __init__(self, snapshots: dict[str, SecurityResolver]) -> None:
        self.snapshots = snapshots

    def load(self, version: Any) -> SecurityResolver:
        return self.snapshots[str(version)]


def _resolver(day: int, *, sid: str = "SEC-000001") -> SecurityResolver:
    start = datetime(2025, 1, day, tzinfo=UTC)
    return SecurityResolver(
        [SecurityRecord(SecurityId(sid), SecurityType.COMMON_STOCK, start)],
        listings=[
            ListingRecord(
                SecurityId(sid),
                "TSLA",
                "XNAS",
                ValidityPeriod(start, start.replace(day=day + 1)),
                start,
                "test",
            )
        ],
    )


def _plan() -> dict[str, Any]:
    return {
        "evaluation_session": "2025-01-13",
        "initialization_sessions": ["2025-01-03"],
        "security_master_snapshot_version": "00000000-0000-0000-0000-000000000013",
        "source_daily_datasets": [
            {"session": "2025-01-03", "snapshot_version": "00000000-0000-0000-0000-000000000003"}
        ],
        "members": [{"ticker": "TSLA", "mic": "XNAS", "security_id": "SEC-000001"}],
    }


def test_resolves_both_dated_sessions() -> None:
    repo = FakeRepository(
        {
            "00000000-0000-0000-0000-000000000003": _resolver(3),
            "00000000-0000-0000-0000-000000000013": _resolver(13),
        }
    )
    resolver = build_minute_pilot_resolver(_plan(), repo)  # type: ignore[arg-type]
    assert (
        str(
            resolver.resolve_ticker(
                ticker="TSLA",
                mic="XNAS",
                valid_at=datetime(2025, 1, 3, 15, tzinfo=UTC),
                known_at=datetime.now(UTC),
            )
        )
        == "SEC-000001"
    )


def test_rejects_identity_change() -> None:
    repo = FakeRepository(
        {
            "00000000-0000-0000-0000-000000000003": _resolver(3, sid="SEC-000002"),
            "00000000-0000-0000-0000-000000000013": _resolver(13),
        }
    )
    with pytest.raises(ValueError, match="Reference identity changed"):
        build_minute_pilot_resolver(_plan(), repo)  # type: ignore[arg-type]


def test_missing_dated_snapshot() -> None:
    plan = _plan()
    plan["source_daily_datasets"] = []
    with pytest.raises(ValueError, match="No pinned reference snapshot"):
        build_minute_pilot_resolver(plan, FakeRepository({}))  # type: ignore[arg-type]


def test_accepts_historical_exchange_change() -> None:
    """Same permanent ID may move from NYSE to Nasdaq."""
    before = datetime(2024, 10, 15, 14, 30, tzinfo=UTC)
    after = datetime(2024, 12, 2, 14, 30, tzinfo=UTC)
    sid = SecurityId("SEC-000001")
    historical = SecurityResolver(
        [SecurityRecord(sid, SecurityType.COMMON_STOCK, before)],
        listings=[ListingRecord(sid, "PLTR", "XNYS", ValidityPeriod(before), before, "test")],
    )
    current = SecurityResolver(
        [SecurityRecord(sid, SecurityType.COMMON_STOCK, before)],
        listings=[ListingRecord(sid, "PLTR", "XNAS", ValidityPeriod(after), after, "test")],
    )
    member = {"ticker": "PLTR", "mic": "XNAS", "security_id": "SEC-000001"}
    assert (
        resolve_historical_member(historical, member, valid_at=before, known_at=datetime.now(UTC))
        == sid
    )
    assert (
        resolve_historical_member(current, member, valid_at=after, known_at=datetime.now(UTC))
        == sid
    )


def test_rejects_historical_identity_change() -> None:
    opening = datetime(2024, 10, 15, 14, 30, tzinfo=UTC)
    wrong = SecurityId("SEC-000002")
    snapshot = SecurityResolver(
        [SecurityRecord(wrong, SecurityType.COMMON_STOCK, opening)],
        listings=[ListingRecord(wrong, "PLTR", "XNYS", ValidityPeriod(opening), opening, "test")],
    )
    with pytest.raises(ValueError, match="Reference identity changed"):
        resolve_historical_member(
            snapshot,
            {"ticker": "PLTR", "mic": "XNAS", "security_id": "SEC-000001"},
            valid_at=opening,
            known_at=datetime.now(UTC),
        )
