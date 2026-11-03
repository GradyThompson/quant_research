"""Massive US-equity regular-session minute aggregate adapter.

Vendor timestamps denote minute START; ProviderBar.timestamp denotes minute END.
Availability is a conservative earliest bound, not confirmed publication time.
"""

from datetime import UTC, datetime, timedelta
from math import isfinite

import pandas as pd

from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderCapabilityError,
    ProviderDataError,
)
from quant_research.data.schemas import BarFrequency


class MassiveMinuteProvider(MassiveDailyProvider):
    """Fetch unadjusted regular-session US-equity one-minute OHLCV bars."""

    def fetch_bars(self, request: BarRequest) -> BarResponse:
        if request.frequency != BarFrequency.MINUTE:
            raise ProviderCapabilityError("This adapter supports minute bars only")
        if request.instrument.identifier_kind != InstrumentKind.TICKER:
            raise ProviderCapabilityError("Massive aggregates require a ticker")
        if request.instrument.mic not in (None, "XNYS", "XNAS"):
            raise ProviderCapabilityError("Only US NYSE/Nasdaq listings are supported")

        # Vendor filters by minute START/date, whereas TimeRange filters by
        # minute END. One minute of padding is sufficient at UTC midnight.
        # Do NOT request adjacent sessions for a single-session research read:
        # an adjacent session might be part of a protected final holdout.
        start_date = (request.period.start.astimezone(UTC) - timedelta(minutes=1)).date()
        end_date = request.period.end.astimezone(UTC).date()
        try:
            raw = self._client.list_aggs(
                ticker=request.instrument.identifier,
                multiplier=1,
                timespan="minute",
                from_=start_date.isoformat(),
                to=end_date.isoformat(),
                adjusted=False,
                sort="asc",
                limit=50000,
            )
            bars: list[ProviderBar] = []
            for item in raw:
                start_ms = int(item.timestamp)
                if start_ms % 60000 != 0:
                    raise ProviderDataError("Massive minute timestamp is not minute-aligned")
                bar_start = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
                bar_end = bar_start + timedelta(minutes=1)
                if not request.period.contains(bar_end):
                    continue
                # Exclude extended hours, weekends and holidays. NYSE calendar
                # also models scheduled early closes and daylight saving time.
                if not self._calendar.is_open_on_minute(pd.Timestamp(bar_start)):
                    continue
                volume: int | None = None
                if item.volume is not None:
                    raw_volume = float(item.volume)
                    if not isfinite(raw_volume) or raw_volume < 0 or not raw_volume.is_integer():
                        raise ProviderDataError("Invalid or fractional Massive minute volume")
                    volume = int(raw_volume)
                try:
                    bars.append(
                        ProviderBar(
                            timestamp=bar_end,
                            available_at=bar_end,  # lower bound; apply lag downstream
                            open=float(item.open),
                            high=float(item.high),
                            low=float(item.low),
                            close=float(item.close),
                            volume=volume,
                            currency="USD",
                        )
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ProviderDataError("Invalid Massive minute aggregate bar") from exc
            bars.sort(key=lambda bar: bar.timestamp)
            return BarResponse(
                provider=self.name,
                request=request,
                retrieved_at=datetime.now(UTC),
                bars=tuple(bars),
            )
        except ProviderDataError:
            raise
        except (ValueError, OverflowError) as exc:
            raise ProviderDataError("Invalid Massive minute aggregate response") from exc
        except Exception as exc:
            raise ProviderDataError("Massive minute aggregate request failed") from exc
