from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import pytest

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.ingestion.daily import DailyIngestionResult
from quant_research.data.ingestion.massive_pilot import daily_request, ingest_pilot, write_receipt
from quant_research.data.storage.models import DatasetMetadata


class FakeIngestor:
    def __init__(self) -> None:
        self.calls = []

    def ingest(self, request, known_at, dataset_id=None):
        self.calls.append((request, known_at, dataset_id))
        return DailyIngestionResult(
            SecurityId("SEC-000001"),
            DatasetMetadata(
                dataset_id=dataset_id,
                version=DatasetVersion(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")),
                created_at=known_at,
                row_count=21,
                schema_fingerprint="fingerprint",
                content_hash="hash",
            ),
        )


def test_request_includes_last_calendar_date():
    req = daily_request("AAPL", "XNAS", date(2025, 1, 2), date(2025, 1, 31))
    assert req.period.start == datetime(2025, 1, 2, tzinfo=UTC)
    assert req.period.end == datetime(2025, 2, 1, tzinfo=UTC)


def test_batch_uses_distinct_ids():
    ingestor = FakeIngestor()
    receipts = ingest_pilot(
        ingestor,
        targets=(("AAPL", "XNAS"), ("MSFT", "XNAS")),
        start=date(2025, 1, 2),
        end=date(2025, 1, 31),
        known_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert [x["dataset_id"] for x in receipts] == [
        "market.ohlcv.daily.aapl",
        "market.ohlcv.daily.msft",
    ]
    assert ingestor.calls[0][2] == DatasetId("market.ohlcv.daily.aapl")
    assert len(ingestor.calls) == 2


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="start"):
        daily_request("AAPL", "XNAS", date(2025, 2, 1), date(2025, 1, 1))
    with pytest.raises(ValueError, match="Pilot"):
        daily_request("AAPL", "XTSE", date(2025, 1, 1), date(2025, 1, 2))
    with pytest.raises(ValueError, match="unique"):
        ingest_pilot(
            FakeIngestor(),
            targets=(("AAPL", "XNAS"), ("AAPL", "XNAS")),
            start=date(2025, 1, 1),
            end=date(2025, 1, 2),
            known_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_receipt_write_is_non_overwriting(tmp_path: Path):
    output = tmp_path / "receipt.json"
    write_receipt(
        output,
        snapshot_version=DatasetVersion(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")),
        receipts=[],
    )
    assert '"datasets": []' in output.read_text()
    with pytest.raises(FileExistsError):
        write_receipt(
            output,
            snapshot_version=DatasetVersion(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")),
            receipts=[],
        )
