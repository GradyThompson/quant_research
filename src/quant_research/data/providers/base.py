"""Minimal provider interface; integrations implement this contract."""

from typing import Protocol

from quant_research.data.providers.models import BarRequest, BarResponse


class MarketDataProvider(Protocol):
    @property
    def name(self) -> str:
        """Stable provenance name (e.g. provider + feed)."""
        ...

    def fetch_bars(self, request: BarRequest) -> BarResponse:
        """Return provider bars or raise ProviderDataError/ProviderCapabilityError.

        Must never silently truncate a requested date range or change the
        requested adjustment convention. Prices must be raw/unadjusted.
        """
        ...
