"""Bounded manifest and historical identifier tests."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research.data.ingestion.historical_daily_window import ingest_historical_daily_window


class FakeIngestor:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def ingest(self, request, known_at, dataset_id=None):
        self.requests.append(request.instrument.identifier)
        return SimpleNamespace(
            security_id="SEC-1",
            metadata=SimpleNamespace(dataset_id=dataset_id, version="version-a", row_count=1),
        )


def test_historical_ticker_and_resume(tmp_path: Path) -> None:
    ingestor = FakeIngestor()
    options = dict(
        members=[{"security_id": "SEC-1", "ticker": "NEW", "mic": "XNAS"}],
        snapshots={"2025-01-06": "snap-6", "2025-01-07": "snap-7"},
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path,
        make_ingestor=lambda _: ingestor,
        historical_listing=lambda snapshot, _: (
            ("OLD", "XNAS") if snapshot == "snap-6" else ("NEW", "XNAS")
        ),
        known_at=datetime(2025, 2, 1, tzinfo=UTC),
        max_requests=1,
    )
    first = ingest_historical_daily_window(**options)
    assert len(first["results"]) == 1
    assert ingestor.requests == ["OLD"]
    second = ingest_historical_daily_window(**options)
    assert len(second["results"]) == 2
    assert ingestor.requests == ["OLD", "NEW"]
    assert second["results"]["2025-01-06|SEC-1"]["dataset_id"].endswith("old.s20250106")


def test_unlisted_security_is_not_requested(tmp_path: Path) -> None:
    ingestor = FakeIngestor()
    result = ingest_historical_daily_window(
        members=[{"security_id": "SEC-1", "ticker": "NEW", "mic": "XNAS"}],
        snapshots={"2025-01-06": "snap-6"},
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path,
        make_ingestor=lambda _: ingestor,
        historical_listing=lambda *_: None,
        known_at=datetime(2025, 2, 1, tzinfo=UTC),
        max_requests=1,
    )
    assert result["results"]["2025-01-06|SEC-1"]["status"] == "not_listed"
    assert not ingestor.requests


def test_manifest_mismatch_rejected(tmp_path: Path) -> None:
    common = dict(
        members=[{"security_id": "SEC-1", "ticker": "NEW", "mic": "XNAS"}],
        snapshots={"2025-01-06": "snap-6"},
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path,
        make_ingestor=lambda _: FakeIngestor(),
        historical_listing=lambda *_: None,
        known_at=datetime(2025, 2, 1, tzinfo=UTC),
        max_requests=1,
    )
    ingest_historical_daily_window(**common)
    with pytest.raises(ValueError, match="Manifest input mismatch"):
        ingest_historical_daily_window(**{**common, "snapshots": {"2025-01-06": "different"}})
