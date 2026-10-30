"""Live smoke test of one regular US-equity minute-aggregate session."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import TimeRange
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency


def verify_session(response: BarResponse, session_date: date) -> None:
    """Verify temporal and OHLCV invariants without assuming full coverage."""
    if not response.bars:
        raise ValueError("No minute bars returned; check API access and requested date")

    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(session_date)
    session_open = calendar.session_open(session).to_pydatetime()
    session_close = calendar.session_close(session).to_pydatetime()
    seen = set()

    for bar in response.bars:
        if not session_open < bar.timestamp <= session_close:
            raise ValueError(f"Bar outside regular session: {bar.timestamp}")
        if bar.timestamp in seen:
            raise ValueError(f"Duplicate minute end: {bar.timestamp}")
        seen.add(bar.timestamp)
        if bar.available_at < bar.timestamp:
            raise ValueError("Availability precedes minute end")
        if any(getattr(bar, field) is None for field in ("open", "high", "low", "close")):
            raise ValueError("Missing OHLC price in minute bar")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--mic", default="XNAS", choices=("XNAS", "XNYS"))
    parser.add_argument("--date", type=date.fromisoformat, default=date(2025, 1, 8))
    args = parser.parse_args()

    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(args.date)
    if not calendar.is_session(session):
        parser.error(f"{args.date} is not an XNYS trading session")

    start = calendar.session_open(session).to_pydatetime()
    end = calendar.session_close(session).to_pydatetime()
    # Half-open request: include the minute ending exactly at session close.
    request = BarRequest(
        instrument=ProviderInstrument(
            identifier=args.ticker,
            identifier_kind=InstrumentKind.TICKER,
            mic=args.mic,
        ),
        frequency=BarFrequency.MINUTE,
        period=TimeRange(start=start, end=end + timedelta(microseconds=1)),
    )
    response = MassiveMinuteProvider().fetch_bars(request)
    verify_session(response, args.date)

    print(f"Ticker: {args.ticker} ({args.mic}); session: {args.date}")
    print(f"Regular session UTC: {start.isoformat()} to {end.isoformat()}")
    print(f"Minute bars received: {len(response.bars)}")
    print(f"First minute END: {response.bars[0].timestamp.isoformat()}")
    print(f"Last minute END:  {response.bars[-1].timestamp.isoformat()}")
    print(f"First OHLCV: {response.bars[0]}")
    print(f"Last OHLCV:  {response.bars[-1]}")
    print("PASS: returned minute bars satisfy session, timestamp and OHLC checks")
    if len(response.bars) != 390:
        print("NOTE: Bar count differs from 390; missing/no-trade minutes may be legitimate.")


if __name__ == "__main__":
    main()
