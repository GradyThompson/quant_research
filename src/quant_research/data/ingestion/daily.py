"""Conservative daily-bar ingestion: provider -> security master -> Arrow -> store.

This is a snapshot ingestion pipeline, not an incremental ingestion scheduler.
"""

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
class DailyIngestionResult:
    security_id: SecurityId
    metadata: DatasetMetadata


class DailyBarIngestor:
    """Ingest one ticker's unadjusted bars into an immutable dataset snapshot.

    The known_at timestamp controls reference-data visibility. This is *not*
    necessarily the market decision timestamp. Historical experiments must
    additionally use point-in-time security-master snapshots and data versions.
    """

    def __init__(
        self,
        *,
        provider: MarketDataProvider,
        resolver: SecurityResolver,
        store: DatasetStore,
        publication_lag: timedelta = timedelta(minutes=15),
    ) -> None:
        if publication_lag <= timedelta(0):
            raise ValueError("publication_lag must be strictly positive")
        self._provider = provider
        self._resolver = resolver
        self._store = store
        self._publication_lag = publication_lag

    def ingest(
        self,
        request: BarRequest,
        known_at: datetime,
        dataset_id: DatasetId | None = None,
    ) -> DailyIngestionResult:
        if dataset_id is None:
            dataset_id = DatasetId("market.ohlcv.daily")
        if request.frequency != BarFrequency.DAILY:
            raise ValueError("DailyBarIngestor only accepts daily bars")
        if request.instrument.identifier_kind != InstrumentKind.TICKER:
            raise ValueError("DailyBarIngestor requires a ticker request")
        if request.instrument.mic is None:
            raise ValueError("An explicit MIC is required for safe ticker resolution")
        if known_at.tzinfo is None or known_at.utcoffset() != timedelta(0):
            raise ValueError("known_at must be a UTC-aware datetime")

        response = self._provider.fetch_bars(request)
        if response.request != request:
            raise DataValidationError("Provider response request does not match")
        if response.provider != self._provider.name:
            raise DataValidationError("Provider response identity does not match")
        if response.retrieved_at.tzinfo is None or response.retrieved_at.utcoffset() != timedelta(
            0
        ):
            raise DataValidationError("Provider retrieval timestamp must be UTC")
        if not response.bars:
            raise DataValidationError("No bars returned; nothing published")

        rows: list[dict[str, object]] = []
        resolved: SecurityId | None = None
        for bar in response.bars:
            sid = self._resolver.resolve_ticker(
                ticker=request.instrument.identifier,
                mic=request.instrument.mic,
                valid_at=bar.timestamp,
                known_at=known_at,
            )
            if resolved is not None and sid != resolved:
                raise DataValidationError(
                    "Ticker maps to multiple securities within requested bars"
                )
            resolved = sid
            # Provider supplies session-close lower bound, not verified publication.
            # Explicit research convention: no sooner than 15 minutes after close.
            available_at = max(bar.available_at, bar.timestamp + self._publication_lag)
            if response.retrieved_at < available_at:
                raise DataValidationError("Bar not yet available at provider retrieval time")
            rows.append(
                {
                    "security_id": str(sid),
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
        assert resolved is not None
        try:
            table = pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))
        except (pa.ArrowException, ValueError, TypeError) as exc:
            raise DataValidationError(
                "Cannot normalize provider bars into canonical OHLCV"
            ) from exc
        OHLCVValidator(frequency=BarFrequency.DAILY).validate(table)
        metadata = self._store.write(dataset_id, table)
        return DailyIngestionResult(security_id=resolved, metadata=metadata)
