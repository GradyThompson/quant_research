"""Integration tests for exact-version security master persistence."""

from datetime import UTC, datetime

import pytest

from quant_research.core import SecurityId
from quant_research.data.storage import ParquetDatasetStore
from quant_research.security_master import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityMasterRepository,
    SecurityRecord,
    SecurityResolver,
    SecurityType,
    ValidityPeriod,
)


def t(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=UTC)


def example() -> SecurityResolver:
    sid = SecurityId("SEC-000001")
    return SecurityResolver(
        [SecurityRecord(sid, SecurityType.COMMON_STOCK, t(2020), "ISS-01")],
        [
            IdentifierRecord(
                sid, IdentifierType.ISIN, "US0378331005", ValidityPeriod(t(2020)), t(2021), "test"
            )
        ],
        [ListingRecord(sid, "ABC", "XNAS", ValidityPeriod(t(2020)), t(2021), "test")],
    )


def test_round_trip(tmp_path):
    repo = SecurityMasterRepository(ParquetDatasetStore(tmp_path))
    snapshot = repo.save(example())
    loaded = repo.load(snapshot.version)
    assert loaded == example()
    assert loaded.resolve_ticker(
        ticker="ABC", mic="XNAS", valid_at=t(2022), known_at=t(2022)
    ) == SecurityId("SEC-000001")
    assert loaded.resolve_identifier(
        identifier_type=IdentifierType.ISIN,
        identifier_value="US0378331005",
        valid_at=t(2022),
        known_at=t(2022),
    ) == SecurityId("SEC-000001")
    assert repo.list_snapshots() == [snapshot]


def test_empty_round_trip(tmp_path):
    repo = SecurityMasterRepository(ParquetDatasetStore(tmp_path))
    snap = repo.save(SecurityResolver([]))
    assert repo.load(snap.version) == SecurityResolver([])


def test_immutable_snapshots(tmp_path):
    repo = SecurityMasterRepository(ParquetDatasetStore(tmp_path))
    first = repo.save(example())
    second = repo.save(SecurityResolver([]))
    assert first.version != second.version
    assert repo.load(first.version) == example()
    assert repo.load(second.version) == SecurityResolver([])


def test_failure_does_not_publish_snapshot(tmp_path, monkeypatch):
    store = ParquetDatasetStore(tmp_path)
    repo = SecurityMasterRepository(store)
    old = repo.save(example())
    original = store.write
    calls = 0

    def fail_third(dataset_id, table):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated failure")
        return original(dataset_id, table)

    monkeypatch.setattr(store, "write", fail_third)
    with pytest.raises(OSError, match="simulated failure"):
        repo.save(example())
    assert repo.list_snapshots() == [old]
    assert repo.load(old.version) == example()
