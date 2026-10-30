from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from quant_research.security_master.massive_bootstrap import bootstrap_massive_security_master


class FakeClient:
    def __init__(self, *, changed: bool = False) -> None:
        self.changed = changed
        self.calls: list[dict] = []

    def list_tickers(self, **kwargs):
        self.calls.append(kwargs)
        figi = (
            "BBG000B9XRY4" if not self.changed or kwargs["date"] == "2025-01-02" else "BBG000OTHER"
        )
        return [
            SimpleNamespace(
                ticker=kwargs["ticker"],
                primary_exchange="XNAS",
                locale="us",
                type="CS",
                share_class_figi=figi,
            )
        ]


def test_bootstrap_with_confirmed_continuity():
    client = FakeClient()
    resolver = bootstrap_massive_security_master(
        client,
        [("AAPL", "XNAS")],
        valid_from=date(2025, 1, 2),
        valid_through=date(2025, 1, 3),
        recorded_at=datetime(2026, 10, 30, tzinfo=UTC),
        historical_continuity_confirmed=True,
    )
    assert len(resolver.securities) == 1
    assert len(client.calls) == 2
    assert (
        resolver.resolve_ticker(
            ticker="AAPL",
            mic="XNAS",
            valid_at=datetime(2025, 1, 2, 21, tzinfo=UTC),
            known_at=datetime(2026, 10, 30, tzinfo=UTC),
        )
        == resolver.securities[0].security_id
    )


def test_requires_explicit_historical_attestation():
    with pytest.raises(ValueError, match="continuity confirmation"):
        bootstrap_massive_security_master(
            FakeClient(),
            [("AAPL", "XNAS")],
            valid_from=date(2025, 1, 2),
            valid_through=date(2025, 1, 3),
            recorded_at=datetime.now(UTC),
        )


def test_refuses_identity_changes():
    with pytest.raises(ValueError, match="Identity changed"):
        bootstrap_massive_security_master(
            FakeClient(changed=True),
            [("AAPL", "XNAS")],
            valid_from=date(2025, 1, 2),
            valid_through=date(2025, 1, 3),
            recorded_at=datetime.now(UTC),
            historical_continuity_confirmed=True,
        )


def test_refuses_duplicates():
    with pytest.raises(ValueError, match="unique"):
        bootstrap_massive_security_master(
            FakeClient(),
            [("AAPL", "XNAS"), ("AAPL", "XNAS")],
            valid_from=date(2025, 1, 2),
            valid_through=date(2025, 1, 3),
            recorded_at=datetime.now(UTC),
            historical_continuity_confirmed=True,
        )
