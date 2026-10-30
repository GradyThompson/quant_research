"""Offline tests of minute ingestion using public synthetic bars."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from quant_research.core import DatasetId, SecurityId, TimeRange
from quant_research.data.ingestion.minute import MinuteBarIngestor
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency
from quant_research.exceptions import DataValidationError

START = datetime(2025, 1, 8, 14, 30, tzinfo=UTC)
KNOWN = datetime(2026, 10, 30, tzinfo=UTC)
DATASET_ID = DatasetId("market.ohlcv.minute.aapl")


def request(frequency: BarFrequency = BarFrequency.MINUTE) -> BarRequest:
    return BarRequest(
        instrument=ProviderInstrument("AAPL", InstrumentKind.TICKER, "XNAS"),
        frequency=frequency,
        period=TimeRange(START, START + timedelta(minutes=3)),
    )


def make_ingestor(*, bars: tuple[ProviderBar, ...] | None = None):
    requested = request()
    if bars is None:
        bars = (
            ProviderBar(
                timestamp=START + timedelta(minutes=1),
                available_at=START + timedelta(minutes=1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000,
                currency="USD",
            ),
        )
    provider = Mock()
    provider.name = "massive"
    provider.fetch_bars.return_value = BarResponse(
        provider="massive", request=requested, retrieved_at=KNOWN, bars=bars
    )
    resolver = Mock()
    resolver.resolve_ticker.return_value = SecurityId("SEC-000001")
    store = Mock()
    store.write.return_value = Mock(dataset_id=DATASET_ID, row_count=len(bars))
    return MinuteBarIngestor(provider=provider, resolver=resolver, store=store), store


def test_ingest_writes_canonical_minute_schema_and_assumed_availability() -> None:
    ingestor, store = make_ingestor()
    result = ingestor.ingest(request(), known_at=KNOWN, dataset_id=DATASET_ID)
    assert result.security_id == SecurityId("SEC-000001")
    dataset_id, table = store.write.call_args.args
    assert dataset_id == DATASET_ID
    assert table.schema.metadata[b"frequency"] == b"1m"
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["timestamp"] == START + timedelta(minutes=1)
    assert row["available_at"] == START + timedelta(minutes=2)
    assert row["volume"] == 1000


def test_rejects_daily_request_before_fetch() -> None:
    ingestor, store = make_ingestor()
    with pytest.raises(ValueError, match="minute bars"):
        ingestor.ingest(request(BarFrequency.DAILY), known_at=KNOWN, dataset_id=DATASET_ID)
    store.write.assert_not_called()


def test_rejects_no_bars_without_writing() -> None:
    ingestor, store = make_ingestor(bars=())
    with pytest.raises(DataValidationError, match="No minute bars"):
        ingestor.ingest(request(), known_at=KNOWN, dataset_id=DATASET_ID)
    store.write.assert_not_called()


def test_rejects_bad_known_at_without_writing() -> None:
    ingestor, store = make_ingestor()
    with pytest.raises(ValueError, match="UTC-aware"):
        ingestor.ingest(request(), known_at=KNOWN.replace(tzinfo=None), dataset_id=DATASET_ID)
    store.write.assert_not_called()
