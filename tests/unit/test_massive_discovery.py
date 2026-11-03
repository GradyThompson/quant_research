"""Tests for point-in-time bounded Massive reference discovery."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from quant_research.security_master.massive_discovery import discover_massive_common_stocks


class FakeClient:
    def __init__(self, rows: dict[bool, list[object]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def list_tickers(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        return self.rows.get(bool(kwargs["active"]), [])


def _row(ticker: str, figi: str | None) -> object:
    return SimpleNamespace(
        ticker=ticker,
        primary_exchange="XNAS",
        locale="us",
        type="CS",
        share_class_figi=figi,
    )


def test_discovers_active_inactive_and_limits_validity_to_one_date() -> None:
    now = datetime(2026, 10, 31, tzinfo=UTC)
    client = FakeClient({True: [_row("AAA", "FIGI_A")], False: [_row("OLD", "FIGI_B")]})
    resolver, report = discover_massive_common_stocks(
        client,
        as_of=date(2025, 1, 13),
        recorded_at=now,
    )
    assert report.included == 2
    assert [call["active"] for call in client.calls] == [True, False]
    assert all(r.recorded_at == now for r in resolver.listings)
    assert all(r.validity.valid_to == datetime(2025, 1, 14, tzinfo=UTC) for r in resolver.listings)
    assert resolver.listings[0].security_id != resolver.listings[1].security_id


def test_invalid_figi_is_reported_and_duplicate_rows_are_deduplicated() -> None:
    client = FakeClient({True: [_row("AAA", "FIGI_A"), _row("AAA", "FIGI_A"), _row("NO", None)]})
    _, report = discover_massive_common_stocks(
        client,
        as_of=date(2025, 1, 13),
        recorded_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert (report.included, report.missing_identity, report.duplicate_rows) == (1, 1, 1)


def test_conflicting_identity_is_quarantined() -> None:
    client = FakeClient(
        {True: [_row("AAA", "FIGI_A"), _row("AAA", "FIGI_B"), _row("OK", "FIGI_OK")]}
    )
    resolver, report = discover_massive_common_stocks(
        client,
        as_of=date(2025, 1, 13),
        recorded_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert report.included == 1
    assert report.conflicting_figis == 2
    assert report.conflicting_listings == 1
    assert report.excluded_conflict_rows == 2
    assert [r.ticker for r in resolver.listings] == ["OK"]


def test_figi_with_multiple_listings_is_quarantined() -> None:
    client = FakeClient(
        {True: [_row("AAA", "FIGI_A"), _row("BBB", "FIGI_A"), _row("OK", "FIGI_OK")]}
    )
    resolver, report = discover_massive_common_stocks(
        client,
        as_of=date(2025, 1, 13),
        recorded_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert report.conflicting_figis == 1
    assert report.excluded_conflict_rows == 2
    assert [r.ticker for r in resolver.listings] == ["OK"]


def test_empty_inventory_fails_closed() -> None:
    with pytest.raises(ValueError, match="No identifiable"):
        discover_massive_common_stocks(
            FakeClient({}),
            as_of=date(2025, 1, 13),
            recorded_at=datetime(2026, 10, 31, tzinfo=UTC),
        )
