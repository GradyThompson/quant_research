"""Reproducible offline daily-bar backtest using pinned dataset snapshots.

This loader intentionally rejects mixed timestamp/schema conventions and does
not assume the latest version of any dataset. It never contacts a provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pyarrow as pa

from quant_research.backtesting.engine import (
    BacktestResult,
    DailyBacktestEngine,
    OrderPolicy,
)
from quant_research.backtesting.execution import DailyBar
from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetStore
from quant_research.data.validation import OHLCVValidator
from quant_research.trading.contracts import Instrument
from quant_research.trading.ledger import PortfolioLedger


@dataclass(frozen=True, slots=True)
class DatasetPin:
    dataset_id: DatasetId
    version: DatasetVersion


def read_pinned_daily_data(
    store: DatasetStore,
    pins: Sequence[DatasetPin],
) -> tuple[tuple[DailyBar, ...], dict[tuple[str, datetime], datetime]]:
    """Load immutable canonical raw daily bars, rejecting duplicate keys.

    Only historical snapshots supplied by the caller are read. No implicit
    version discovery or provider access occurs. Availability is NOT verified
    against vendor publication: this must be checked at ingestion.
    """
    if not pins:
        raise ValueError("at least one dataset pin is required")
    expected_schema = ohlcv_schema(frequency=BarFrequency.DAILY)
    validator = OHLCVValidator(frequency=BarFrequency.DAILY)
    result: list[DailyBar] = []
    seen: set[tuple[str, datetime]] = set()
    availability: dict[tuple[str, datetime], datetime] = {}
    for pin in pins:
        table: pa.Table = store.read(pin.dataset_id, pin.version)
        validator.validate(table)
        if not table.schema.equals(expected_schema, check_metadata=True):
            raise ValueError("only canonical raw daily OHLCV datasets are supported")
        for row in table.to_pylist():
            key = (row["security_id"], row["timestamp"])
            if key in seen:
                raise ValueError("duplicate security/time across pinned datasets")
            seen.add(key)
            if row["available_at"] < row["timestamp"]:
                raise ValueError("bar availability precedes bar close")
            availability[key] = row["available_at"]
            values = (row["open"], row["high"], row["low"], row["close"])
            if any(value is None for value in values):
                raise ValueError("backtest requires non-null OHLC prices")
            result.append(
                DailyBar(
                    security_id=row["security_id"],
                    timestamp=row["timestamp"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=row["volume"],
                )
            )
    result.sort(key=lambda bar: (bar.timestamp, bar.security_id))
    return tuple(result), availability


def read_pinned_daily_bars(
    store: DatasetStore,
    pins: Sequence[DatasetPin],
) -> tuple[DailyBar, ...]:
    """Return bars only; use `read_pinned_daily_data` for decision timing."""
    return read_pinned_daily_data(store, pins)[0]


def run_pinned_backtest(
    *,
    store: DatasetStore,
    pins: Sequence[DatasetPin],
    instruments: Mapping[str, Instrument],
    initial_ledger: PortfolioLedger,
    policy: OrderPolicy,
    engine: DailyBacktestEngine | None = None,
) -> BacktestResult:
    """Run a caller-provided *local* policy over pinned historical data."""
    bars, availability = read_pinned_daily_data(store, pins)
    return (engine or DailyBacktestEngine()).run(
        bars=bars,
        instruments=instruments,
        initial_ledger=initial_ledger,
        policy=policy,
        availability=availability,
    )
