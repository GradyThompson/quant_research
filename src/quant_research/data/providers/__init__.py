"""Provider-independent market data ingestion contracts."""

from quant_research.data.providers.base import MarketDataProvider
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderCapabilityError,
    ProviderDataError,
    ProviderInstrument,
)

__all__ = [
    "MarketDataProvider",
    "BarRequest",
    "BarResponse",
    "InstrumentKind",
    "ProviderBar",
    "ProviderCapabilityError",
    "ProviderDataError",
    "ProviderInstrument",
]
