"""Provider contract tests, without credentials or network access."""

from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import TimeRange
from quant_research.data.providers import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    MarketDataProvider,
    ProviderBar,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency


def request() -> BarRequest:
    return BarRequest(
        instrument=ProviderInstrument("TEST", InstrumentKind.TICKER, "XNAS"),
        frequency=BarFrequency.DAILY,
        period=TimeRange(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 4, tzinfo=UTC),
        ),
    )


def bar(day: int = 2) -> ProviderBar:
    timestamp = datetime(2025, 1, day, 21, tzinfo=UTC)
    return ProviderBar(
        timestamp=timestamp,
        available_at=timestamp + timedelta(seconds=5),
        open=100.0,
        high=103.0,
        low=99.0,
        close=102.0,
        volume=1000,
        currency="USD",
    )


class FakeProvider:
    name = "fake/raw"

    def fetch_bars(self, req: BarRequest) -> BarResponse:
        return BarResponse(self.name, req, datetime(2025, 1, 5, tzinfo=UTC), (bar(),))


def test_fake_provider_satisfies_contract():
    provider: MarketDataProvider = FakeProvider()
    response = provider.fetch_bars(request())
    assert response.provider == "fake/raw"
    assert len(response.bars) == 1


def test_provider_instrument_validates_mic():
    with pytest.raises(ValueError, match="mic"):
        ProviderInstrument("TEST", InstrumentKind.TICKER, "nasdaq")


def test_bar_rejects_temporal_leakage():
    b = bar()
    with pytest.raises(ValueError, match="available_at"):
        ProviderBar(
            timestamp=b.timestamp,
            available_at=b.timestamp - timedelta(seconds=1),
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
            currency=b.currency,
        )


def test_response_rejects_duplicate_bars():
    with pytest.raises(ValueError, match="unique"):
        BarResponse("fake", request(), datetime(2025, 1, 5, tzinfo=UTC), (bar(), bar()))


def test_response_rejects_bars_outside_requested_window():
    with pytest.raises(ValueError, match="requested period"):
        BarResponse("fake", request(), datetime(2025, 1, 5, tzinfo=UTC), (bar(4),))


def test_response_rejects_unsorted_bars():
    with pytest.raises(ValueError, match="strictly increasing"):
        BarResponse("fake", request(), datetime(2025, 1, 5, tzinfo=UTC), (bar(3), bar(2)))


def test_bar_rejects_bad_prices():
    b = bar()
    with pytest.raises(ValueError, match="high"):
        ProviderBar(
            timestamp=b.timestamp,
            available_at=b.available_at,
            open=b.open,
            high=90,
            low=b.low,
            close=b.close,
            volume=b.volume,
            currency=b.currency,
        )
