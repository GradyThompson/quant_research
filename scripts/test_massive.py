from datetime import UTC, datetime

from quant_research.core import TimeRange
from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.providers.models import (
    BarRequest,
    InstrumentKind,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency


def main() -> None:
    provider = MassiveDailyProvider()

    request = BarRequest(
        instrument=ProviderInstrument(
            identifier="AAPL",
            identifier_kind=InstrumentKind.TICKER,
            mic="XNAS",
        ),
        frequency=BarFrequency.DAILY,
        period=TimeRange(
            start=datetime(2025, 1, 2, tzinfo=UTC),
            end=datetime(2025, 1, 11, tzinfo=UTC),
        ),
    )

    response = provider.fetch_bars(request)

    print(f"Provider: {response.provider}")
    print(f"Bars received: {len(response.bars)}")

    for bar in response.bars:
        print(f"{bar.timestamp.date()} | Close: ${bar.close:.2f} | Volume: {bar.volume:,}")


if __name__ == "__main__":
    main()
