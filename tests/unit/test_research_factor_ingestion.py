"""Offline unit tests for strict, resumable ETF factor pinning."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import pytest

# The receipt orchestration is intentionally standard-library-only. Stub the
# existing atomic writer so the tests do not require a Parquet installation.
from quant_research.data.research_factor import (
    FactorIdentity,
    FactorPin,
    ingest_research_factor,
    requested_factor_days,
)


@pytest.fixture(autouse=True)
def fake_atomic_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("quant_research.data.ingestion.minute_pilot_window")

    def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".pending")
        if temp.exists():
            raise FileExistsError("Stale temp receipt")
        temp.write_text(json.dumps(payload), encoding="utf-8")
        temp.replace(path)

    module._write_json_atomic = write_json_atomic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def permitted_days() -> tuple[date, ...]:
    return tuple(date(2024, 12, 2) + timedelta(days=i) for i in range(3))


def run(
    path: Path,
    days: tuple[date, ...],
    *,
    lookup,
    ingest,
    verify,
    max_sessions: int = 5,
    dry_run: bool = False,
    digest: str | None = None,
) -> dict[str, object]:
    return ingest_research_factor(
        receipt_path=path,
        asset_receipt_sha256=digest or hashlib.sha256(b"asset-receipt").hexdigest(),
        permitted_days=days,
        ticker="QQQ",
        mic="XNAS",
        publication_lag_seconds=60,
        lookup_identity=lookup,
        ingest_session=ingest,
        verify_existing=verify,
        max_sessions=max_sessions,
        dry_run=dry_run,
    )


def test_only_permitted_dates_and_resume(tmp_path: Path, permitted_days: tuple[date, ...]) -> None:
    path = tmp_path / "factor.json"
    lookups: list[date] = []
    ingestions: list[date] = []
    verified: list[str] = []

    def lookup(day: date) -> FactorIdentity:
        lookups.append(day)
        return FactorIdentity("QQQ", "XNAS", "FIGI-QQQ", "SEC-QQQ")

    def ingest(day: date, identity: FactorIdentity) -> FactorPin:
        assert identity.security_id == "SEC-QQQ"
        ingestions.append(day)
        return FactorPin(
            "market.ohlcv.minute.qqq.s" + day.strftime("%Y%m%d"), "version", 390, "first", "last"
        )

    def verify(entry: dict[str, object]) -> None:
        verified.append(str(entry["session"]))

    assert (
        run(path, permitted_days, lookup=lookup, ingest=ingest, verify=verify, dry_run=True)[
            "missing"
        ]
        == 3
    )
    assert lookups == [] and ingestions == [] and verified == []
    assert not path.exists()
    first = run(path, permitted_days, lookup=lookup, ingest=ingest, verify=verify, max_sessions=1)
    assert first["new"] == 1 and first["missing"] == 2
    second = run(path, permitted_days, lookup=lookup, ingest=ingest, verify=verify)
    assert second["new"] == 2 and second["missing"] == 0
    third = run(path, permitted_days, lookup=lookup, ingest=ingest, verify=verify)
    assert third["new"] == 0
    assert lookups == list(permitted_days)
    assert ingestions == list(permitted_days)
    assert len(verified) == 1 + 1 + 2 + 3  # new readbacks, resume, complete
    payload = json.loads(path.read_text())
    assert len(payload["entries"]) == 3
    assert payload["permitted_sessions"] == [d.isoformat() for d in permitted_days]
    assert payload["reference_data_point_in_time_verified"] is False


def test_identity_change_fails_closed(tmp_path: Path, permitted_days: tuple[date, ...]) -> None:
    path = tmp_path / "factor.json"
    attempts: list[date] = []

    def lookup(day: date) -> FactorIdentity:
        return FactorIdentity(
            "QQQ",
            "XNAS",
            "FIGI-ONE" if day == permitted_days[0] else "FIGI-TWO",
            "SEC-ONE" if day == permitted_days[0] else "SEC-TWO",
        )

    def ingest(day: date, identity: FactorIdentity) -> FactorPin:
        attempts.append(day)
        return FactorPin("dataset", "version", 100, "first", "last")

    with pytest.raises(ValueError, match="identity changed"):
        run(path, permitted_days, lookup=lookup, ingest=ingest, verify=lambda _: None)
    assert attempts == [permitted_days[0]]
    assert len(json.loads(path.read_text())["entries"]) == 1


def test_cannot_change_approved_date_set_or_receipt_hash(
    tmp_path: Path, permitted_days: tuple[date, ...]
) -> None:
    path = tmp_path / "factor.json"

    def lookup(day: date) -> FactorIdentity:
        return FactorIdentity("QQQ", "XNAS", "F", "S")

    def ingest(day: date, identity: FactorIdentity) -> FactorPin:
        return FactorPin("data", "version", 390, "first", "last")

    run(path, permitted_days, lookup=lookup, ingest=ingest, verify=lambda _: None, max_sessions=1)
    with pytest.raises(ValueError, match="metadata differs"):
        run(path, permitted_days[:2], lookup=lookup, ingest=ingest, verify=lambda _: None)
    with pytest.raises(ValueError, match="metadata differs"):
        run(
            path,
            permitted_days,
            lookup=lookup,
            ingest=ingest,
            verify=lambda _: None,
            digest="a" * 64,
        )


def test_failed_provider_request_never_pins_day(
    tmp_path: Path, permitted_days: tuple[date, ...]
) -> None:
    path = tmp_path / "factor.json"

    def fail(day: date, identity: FactorIdentity) -> FactorPin:
        raise OSError("Provider unavailable")

    with pytest.raises(OSError, match="Provider unavailable"):
        run(
            path,
            permitted_days,
            lookup=lambda day: FactorIdentity("QQQ", "XNAS", "F", "S"),
            ingest=fail,
            verify=lambda _: None,
        )
    assert json.loads(path.read_text())["entries"] == {}


def test_unapproved_key_rejected_before_dataset_access(
    tmp_path: Path, permitted_days: tuple[date, ...]
) -> None:
    path = tmp_path / "factor.json"
    pin = FactorPin("data", "version", 390, "first", "last")
    run(
        path,
        permitted_days,
        lookup=lambda day: FactorIdentity("QQQ", "XNAS", "F", "S"),
        ingest=lambda day, identity: pin,
        verify=lambda _: None,
        max_sessions=1,
    )
    obj = json.loads(path.read_text())
    obj["entries"]["QQQ:XNAS:2030-01-01"] = obj["entries"]["QQQ:XNAS:2024-12-02"]
    path.write_text(json.dumps(obj))
    reads: list[object] = []
    with pytest.raises(ValueError, match="unapproved dates"):
        run(
            path,
            permitted_days,
            lookup=lambda day: reads.append(day),
            ingest=lambda day, identity: pin,
            verify=lambda entry: reads.append(entry),
        )
    assert not reads


def test_sorted_unique_dates_required() -> None:
    with pytest.raises(ValueError, match="unique sorted"):
        requested_factor_days([{"session": "2024-01-02"}, {"session": "2024-01-02"}])
