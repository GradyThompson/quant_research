"""Massive US-equity daily aggregate adapter.

Provider timestamps mark BAR START. We map daily bars to the XNYS
session close. Availability is conservatively *not* established by the
aggregate API: the response uses session close as a lower bound; downstream
research must apply an explicit post-close availability lag.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderCapabilityError,
    ProviderDataError,
)
from quant_research.data.schemas import BarFrequency


class _AggClient(Protocol):
    def list_aggs(self, **kwargs: Any) -> Iterable[Any]: ...


class MassiveDailyProvider:
    """Read raw US equity daily bars through Massive's REST SDK.

    A client may be injected for offline testing. The adapter does not
    resolve tickers to internal securities or persist results.
    """

    def __init__(self, client: _AggClient | None = None) -> None:
        if client is None:
            import os

            from massive import RESTClient

            api_key = os.environ.get("MASSIVE_API_KEY")
            if not api_key:
                raise ProviderDataError("MASSIVE_API_KEY is not configured")
            client = RESTClient(api_key=api_key)
        self._client = client
        self._calendar = xcals.get_calendar("XNYS")

    @property
    def name(self) -> str:
        return "massive.us_stocks.aggregates"

    def fetch_bars(self, request: BarRequest) -> BarResponse:
        if request.frequency != BarFrequency.DAILY:
            raise ProviderCapabilityError("This adapter supports daily bars only")
        if request.instrument.identifier_kind != InstrumentKind.TICKER:
            raise ProviderCapabilityError("Massive aggregates require a ticker")
        if request.instrument.mic not in (None, "XNYS", "XNAS"):
            raise ProviderCapabilityError("Only US NYSE/Nasdaq listings are supported")

        # Request a superset of UTC dates, then filter on canonical bar-END.
        start = request.period.start.astimezone(UTC).date() - timedelta(days=1)
        end = request.period.end.astimezone(UTC).date() + timedelta(days=1)
        try:
            raw = self._client.list_aggs(
                ticker=request.instrument.identifier,
                multiplier=1,
                timespan="day",
                from_=start.isoformat(),
                to=end.isoformat(),
                adjusted=False,  # vendor default is adjusted: never omit
                sort="asc",
                limit=50000,
            )
            bars: list[ProviderBar] = []
            for item in raw:
                provider_start = datetime.fromtimestamp(int(item.timestamp) / 1000, tz=UTC)
                # The feed's daily timestamp is session-start-like (often midnight
                # NY); use New York date for trading session identification.
                session_date = pd.Timestamp(
                    provider_start.astimezone(ZoneInfo("America/New_York")).date()
                )
                if not self._calendar.is_session(session_date):
                    raise ProviderDataError(
                        f"Unexpected non-XNYS session date: {session_date.date()}"
                    )
                close = self._calendar.session_close(session_date).to_pydatetime()
                if not request.period.contains(close):
                    continue
                vol = item.volume
                if vol is not None:
                    vol_f = float(vol)
                    if not isfinite(vol_f) or vol_f < 0 or not vol_f.is_integer():
                        raise ProviderDataError(
                            "Fractional/invalid vendor volume cannot fit the existing "
                            "integer ProviderBar.volume contract; revise that contract "
                            "before ingesting such bars."
                        )
                    vol = int(vol_f)
                try:
                    bars.append(
                        ProviderBar(
                            timestamp=close,
                            available_at=close,  # earliest bound only, NOT confirmed publication
                            open=float(item.open),
                            high=float(item.high),
                            low=float(item.low),
                            close=float(item.close),
                            volume=vol,
                            currency="USD",
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ProviderDataError("Invalid Massive aggregate bar") from exc
        except ProviderDataError:
            raise
        except Exception as exc:
            raise ProviderDataError("Massive aggregate request failed") from exc
        bars.sort(key=lambda b: b.timestamp)
        try:
            return BarResponse(
                provider=self.name,
                request=request,
                retrieved_at=datetime.now(UTC),
                bars=tuple(bars),
            )
        except ValueError as exc:
            raise ProviderDataError("Massive aggregate response invalid") from exc
