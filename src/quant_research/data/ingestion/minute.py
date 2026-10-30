"""Immutable ingestion of one security's regular-session minute OHLCV."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from quant_research.core import DatasetId, SecurityId
from quant_research.data.providers.base import MarketDataProvider
from quant_research.data.providers.models import BarRequest, InstrumentKind
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetMetadata, DatasetStore
from quant_research.data.validation import OHLCVValidator
from quant_research.exceptions import DataValidationError
from quant_research.security_master.resolver import SecurityResolver


@dataclass(frozen=True, slots=True)
class MinuteIngestionResult:
    """The security and immutable version published by one ingestion."""

    security_id: SecurityId
    metadata: DatasetMetadata


class MinuteBarIngestor:
    """Persist raw minute bars, with an explicit assumed publication delay.

    Publication lag is a research convention, not verified vendor latency.
    It must be revisited before any intraday trading simulation.
    """

    def __init__(
        self,
        *,
        provider: MarketDataProvider,
        resolver: SecurityResolver,
        store: DatasetStore,
        publication_lag: timedelta = timedelta(minutes=1),
    ) -> None:
        if publication_lag <= timedelta(0):
            raise ValueError("publication_lag must be positive")
        self._provider = provider
        self._resolver = resolver
        self._store = store
        self._publication_lag = publication_lag

    def ingest(
        self,
        request: BarRequest,
        *,
        known_at: datetime,
        dataset_id: DatasetId,
    ) -> MinuteIngestionResult:
        if request.frequency != BarFrequency.MINUTE:
            raise ValueError("MinuteBarIngestor requires minute bars")
        if request.instrument.identifier_kind != InstrumentKind.TICKER:
            raise ValueError("MinuteBarIngestor requires a ticker")
        if request.instrument.mic not in ("XNYS", "XNAS"):
            raise ValueError("An explicit XNYS or XNAS MIC is required")
        if known_at.tzinfo is None or known_at.utcoffset() != timedelta(0):
            raise ValueError("known_at must be UTC-aware")

        response = self._provider.fetch_bars(request)
        if response.request != request or response.provider != self._provider.name:
            raise DataValidationError("Provider response identity mismatch")
        if response.retrieved_at.tzinfo is None or response.retrieved_at.utcoffset() != timedelta(
            0
        ):
            raise DataValidationError("Provider retrieval timestamp must be UTC")
        if not response.bars:
            raise DataValidationError("No minute bars returned; nothing published")

        security_id: SecurityId | None = None
        rows: list[dict[str, object]] = []
        for bar in response.bars:
            resolved = self._resolver.resolve_ticker(
                ticker=request.instrument.identifier,
                mic=request.instrument.mic,
                valid_at=bar.timestamp,
                known_at=known_at,
            )
            if security_id is not None and security_id != resolved:
                raise DataValidationError("Ticker resolves to multiple securities in one request")
            security_id = resolved
            available_at = max(bar.available_at, bar.timestamp + self._publication_lag)
            if response.retrieved_at < available_at:
                raise DataValidationError("Bar cannot be ingested before assumed availability")
            rows.append(
                {
                    "security_id": str(resolved),
                    "timestamp": bar.timestamp.astimezone(UTC),
                    "available_at": available_at.astimezone(UTC),
                    "ingested_at": response.retrieved_at.astimezone(UTC),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "currency": bar.currency,
                }
            )
        assert security_id is not None
        try:
            table = pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.MINUTE))
        except (pa.ArrowException, ValueError, TypeError) as exc:
            raise DataValidationError("Cannot normalize minute bars") from exc
        OHLCVValidator(frequency=BarFrequency.MINUTE).validate(table)
        metadata = self._store.write(dataset_id, table)
        return MinuteIngestionResult(security_id=security_id, metadata=metadata)
