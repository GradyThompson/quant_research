"""Offline tests for resumable bounded daily universe ingestion."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from quant_research.data.ingestion.universe_daily import ingest_universe_daily_session


class FakeIngestor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ingest(self, request: object, known_at: datetime, dataset_id: object = None) -> object:
        assert dataset_id is not None
        self.calls.append(str(dataset_id))
        if str(dataset_id).endswith(".bad"):
            raise RuntimeError("no historical bars")
        return SimpleNamespace(
            metadata=SimpleNamespace(dataset_id=dataset_id, version="fixed", row_count=1)
        )


def test_ingestion_bounds_resumes_and_keeps_failures(tmp_path) -> None:
    fake = FakeIngestor()
    members = [
        {"ticker": name, "mic": "XNAS", "security_id": f"SEC-{i + 1000000}"}
        for i, name in enumerate(("GOOD", "BAD", "NEXT"))
    ]
    config = dict(
        ingestor=fake,
        members=members,
        session=date(2025, 1, 10),
        snapshot_version="version-1",
        manifest=tmp_path / "receipt.json",
        known_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    first = ingest_universe_daily_session(**config, max_requests=2)
    assert len(fake.calls) == 2
    assert first["results"]["GOOD"]["status"] == "ok"
    assert first["results"]["BAD"]["status"] == "error"
    second = ingest_universe_daily_session(**config, max_requests=2)
    assert len(fake.calls) == 4  # GOOD skipped, BAD retried, NEXT requested
    assert second["results"]["NEXT"]["status"] == "ok"
    with pytest.raises(ValueError, match="Manifest does not match"):
        ingest_universe_daily_session(**{**config, "session": date(2025, 1, 9)}, max_requests=1)
