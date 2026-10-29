"""Offline integration tests: no Massive credentials or network required."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import DatasetId, SecurityId, TimeRange
from quant_research.data.ingestion import DailyBarIngestor
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency
from quant_research.data.storage import ParquetDatasetStore
from quant_research.exceptions import DataValidationError
from quant_research.security_master import (
    ListingRecord,
    SecurityRecord,
    SecurityResolver,
    SecurityType,
    ValidityPeriod,
)

END = datetime(2025, 1, 2, 21, tzinfo=UTC)
RETRIEVED = END + timedelta(days=2)


class FakeProvider:
    name = "fake.us_stocks"

    def __init__(self, bars: tuple[ProviderBar, ...]):
        self.bars = bars

    def fetch_bars(self, request: BarRequest) -> BarResponse:
        return BarResponse(self.name, request, RETRIEVED, self.bars)


def make_request() -> BarRequest:
    return BarRequest(
        instrument=ProviderInstrument("ABC", InstrumentKind.TICKER, mic="XNYS"),
        frequency=BarFrequency.DAILY,
        period=TimeRange(END - timedelta(days=1), END + timedelta(days=1)),
    )


def make_bar() -> ProviderBar:
    return ProviderBar(END, END, 100.0, 105.0, 98.0, 103.0, 1000, "USD")


def make_resolver() -> SecurityResolver:
    sid = SecurityId("SEC-000001")
    return SecurityResolver(
        securities=[
            SecurityRecord(sid, SecurityType.COMMON_STOCK, datetime(2020, 1, 1, tzinfo=UTC))
        ],
        listings=[
            ListingRecord(
                sid,
                "ABC",
                "XNYS",
                ValidityPeriod(datetime(2020, 1, 1, tzinfo=UTC)),
                datetime(2020, 1, 2, tzinfo=UTC),
                "fixture",
            )
        ],
    )


def make_ingestor(tmp_path, bars=None):
    store = ParquetDatasetStore(tmp_path)
    provider = FakeProvider((make_bar(),) if bars is None else bars)
    return DailyBarIngestor(provider=provider, resolver=make_resolver(), store=store), store


def test_ingestion_roundtrip(tmp_path):
    ingestor, store = make_ingestor(tmp_path)
    result = ingestor.ingest(request=make_request(), known_at=RETRIEVED)
    table = store.read(DatasetId("market.ohlcv.daily"), result.metadata.version)
    row = table.to_pylist()[0]
    assert result.security_id == SecurityId("SEC-000001")
    assert row["security_id"] == "SEC-000001"
    assert row["available_at"] == END + timedelta(minutes=15)
    assert row["ingested_at"] == RETRIEVED
    assert table.schema.metadata[b"price_adjustment"] == b"raw"


def test_no_bars_does_not_publish(tmp_path):
    ingestor, store = make_ingestor(tmp_path, ())
    with pytest.raises(DataValidationError, match="No bars"):
        ingestor.ingest(request=make_request(), known_at=RETRIEVED)
    assert store.list_versions(DatasetId("market.ohlcv.daily")) == []


def test_unknown_historical_ticker_does_not_publish(tmp_path):
    ingestor, store = make_ingestor(tmp_path)
    with pytest.raises(ValueError, match="No security found"):
        ingestor.ingest(request=make_request(), known_at=datetime(2019, 1, 1, tzinfo=UTC))
    assert store.list_versions(DatasetId("market.ohlcv.daily")) == []


def test_unavailable_bar_does_not_publish(tmp_path):
    ingestor, store = make_ingestor(tmp_path)
    ingestor._provider.bars = (replace(make_bar(), available_at=RETRIEVED + timedelta(hours=1)),)
    with pytest.raises(DataValidationError, match="not yet available"):
        ingestor.ingest(request=make_request(), known_at=RETRIEVED)
    assert store.list_versions(DatasetId("market.ohlcv.daily")) == []


def test_request_requires_exchange(tmp_path):
    ingestor, _ = make_ingestor(tmp_path)
    request = replace(make_request(), instrument=ProviderInstrument("ABC", InstrumentKind.TICKER))
    with pytest.raises(ValueError, match="MIC"):
        ingestor.ingest(request=request, known_at=RETRIEVED)
