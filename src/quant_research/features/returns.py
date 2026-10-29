"""Leakage-aware trailing close-to-close return features.

A lookback of N means N prior *observed* bars for the same security.
This does not imply N consecutive exchange sessions. Missing closes reset
warmup; missing sessions require separate coverage validation.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from quant_research.data.schemas import BarFrequency
from quant_research.data.validation import OHLCVValidator


@dataclass(frozen=True, slots=True)
class TrailingReturnFeature:
    lookback: int = 5
    frequency: BarFrequency = BarFrequency.DAILY

    def __post_init__(self) -> None:
        if (
            isinstance(self.lookback, bool)
            or not isinstance(self.lookback, int)
            or self.lookback < 1
        ):
            raise ValueError("lookback must be a positive integer")

    def compute(self, bars: pa.Table) -> pa.Table:
        """Compute past-only returns; output rows are sorted by security/time.

        Input must be canonical raw OHLCV for the selected frequency.
        Output availability matches the completed current input bar.
        This function does not move availability earlier or assert execution
        is possible at that time.
        """
        OHLCVValidator(frequency=self.frequency).validate(bars)
        rows = sorted(bars.to_pylist(), key=lambda row: (row["security_id"], row["timestamp"]))
        histories: dict[str, deque[float]] = {}
        output: list[dict[str, str | datetime | float | None]] = []
        for row in rows:
            sid = row["security_id"]
            history = histories.setdefault(sid, deque(maxlen=self.lookback + 1))
            close = row["close"]
            if close is None:
                history.clear()
                value = None
            else:
                history.append(close)
                value = (close / history[0] - 1.0) if len(history) == self.lookback + 1 else None
            output.append(
                {
                    "security_id": sid,
                    "timestamp": row["timestamp"],
                    "available_at": row["available_at"],
                    "trailing_return": value,
                }
            )
        return pa.Table.from_pylist(output, schema=self.schema())

    def schema(self) -> pa.Schema:
        fields: list[pa.Field] = [
            pa.field("security_id", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("trailing_return", pa.float64()),
        ]

        schema = pa.schema(
            fields,
            metadata={
                b"feature_name": b"trailing_return",
                b"feature_version": b"1",
                b"lookback_observations": str(self.lookback).encode(),
                b"frequency": self.frequency.value.encode(),
                b"price_adjustment": b"raw",
            },
        )
        return schema
