"""Offline tests for the Massive daily-bar adapter."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from quant_research.core import TimeRange
from quant_research.data.providers import BarRequest, InstrumentKind, ProviderInstrument
from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.providers.models import ProviderCapabilityError, ProviderDataError
from quant_research.data.schemas import BarFrequency


@dataclass
class FakeAgg:
    timestamp: int
    open: float = 100.0
    high: float = 105.0
    low: float = 99.0
    close: float = 103.0
    volume: float = 1000.0


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.kwargs = None

    def list_aggs(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.rows)


def request(frequency=BarFrequency.DAILY):
    return BarRequest(
        instrument=ProviderInstrument("AAPL", InstrumentKind.TICKER, "XNAS"),
        frequency=frequency,
        period=TimeRange(datetime(2025, 1, 2, tzinfo=UTC), datetime(2025, 1, 4, tzinfo=UTC)),
    )


def test_daily_bar_maps_to_session_close():
    # Vendor starts the 2025-01-02 daily aggregate at 00:00 NY time.
    start_ms = int(datetime(2025, 1, 2, 5, tzinfo=UTC).timestamp() * 1000)
    client = FakeClient([FakeAgg(start_ms)])
    response = MassiveDailyProvider(client).fetch_bars(request())
    assert len(response.bars) == 1
    assert response.bars[0].timestamp == datetime(2025, 1, 2, 21, tzinfo=UTC)
    assert response.bars[0].available_at == response.bars[0].timestamp
    assert client.kwargs["adjusted"] is False
    assert client.kwargs["timespan"] == "day"


def test_early_close_from_exchange_calendar():
    # Day after Thanksgiving 2025 ends at 13:00 ET, not 16:00.
    start_ms = int(datetime(2025, 11, 28, 5, tzinfo=UTC).timestamp() * 1000)
    client = FakeClient([FakeAgg(start_ms)])
    req = BarRequest(
        request().instrument,
        BarFrequency.DAILY,
        TimeRange(datetime(2025, 11, 28, tzinfo=UTC), datetime(2025, 11, 29, tzinfo=UTC)),
    )
    response = MassiveDailyProvider(client).fetch_bars(req)
    assert response.bars[0].timestamp == datetime(2025, 11, 28, 18, tzinfo=UTC)


def test_reject_minute():
    with pytest.raises(ProviderCapabilityError):
        MassiveDailyProvider(FakeClient([])).fetch_bars(request(BarFrequency.MINUTE))


def test_reject_fractional_volume():
    start_ms = int(datetime(2025, 1, 2, 5, tzinfo=UTC).timestamp() * 1000)
    with pytest.raises(ProviderDataError, match="Fractional"):
        MassiveDailyProvider(FakeClient([FakeAgg(start_ms, volume=0.25)])).fetch_bars(request())


def test_no_bars_is_valid():
    result = MassiveDailyProvider(FakeClient([])).fetch_bars(request())
    assert result.bars == ()
