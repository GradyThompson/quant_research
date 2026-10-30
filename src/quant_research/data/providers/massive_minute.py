"""Scaffold for Massive US-equity minute aggregates.

Retrieval is intentionally not implemented yet. Keeping this separate from
MassiveDailyProvider preserves the existing, verified daily-bar ingestion path.
"""

from quant_research.data.providers.massive import MassiveDailyProvider
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    ProviderCapabilityError,
)
from quant_research.data.schemas import BarFrequency


class MassiveMinuteProvider(MassiveDailyProvider):
    """Minute-bar adapter scaffold; does not issue API requests yet."""

    def fetch_bars(self, request: BarRequest) -> BarResponse:
        """Reject non-minute requests; minute retrieval follows in Step 4.3."""
        if request.frequency != BarFrequency.MINUTE:
            raise ProviderCapabilityError("This adapter supports minute bars only")

        raise NotImplementedError("Minute aggregate retrieval is not yet implemented")
