"""Forward close-to-close labels for *exploratory* prediction research.

Horizon N counts N subsequent observed rows for each security, NOT N
exchange sessions. Labels are not investable/executable returns. All input
bars must meet the raw OHLCV contract; corporate action adjustments are a
separate prerequisite for credible return inference.
"""

from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from quant_research.data.schemas import BarFrequency
from quant_research.data.validation import OHLCVValidator


@dataclass(frozen=True, slots=True)
class ForwardReturnLabel:
    horizon: int = 5
    frequency: BarFrequency = BarFrequency.DAILY

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")

    def compute(self, bars: pa.Table) -> pa.Table:
        """Compute horizon-ahead close returns, aligned to each starting bar.

        A label is defined only if the starting close and *every* close through
        the horizon are non-null; otherwise it is null. This intentionally
        rejects labels crossing missing-close observations, but does not detect
        omitted market sessions. Use exchange-calendar coverage validation later.

        `label_available_at` is the ending bar's availability, not the starting
        bar's availability. Do not merge this field into decision-time features.
        """
        OHLCVValidator(frequency=self.frequency).validate(bars)
        rows = sorted(bars.to_pylist(), key=lambda row: (row["security_id"], row["timestamp"]))
        output: list[dict[str, str | datetime | float | None]] = []
        by_security: dict[str, list[dict]] = {}
        for row in rows:
            by_security.setdefault(row["security_id"], []).append(row)

        for sid, security_rows in by_security.items():
            for idx, start in enumerate(security_rows):
                future_idx = idx + self.horizon
                label_value: float | None = None
                label_end: datetime | None = None
                label_available: datetime | None = None
                if future_idx < len(security_rows):
                    path = security_rows[idx : future_idx + 1]
                    if all(bar["close"] is not None for bar in path):
                        future = security_rows[future_idx]
                        label_value = future["close"] / start["close"] - 1.0
                        label_end = future["timestamp"]
                        label_available = future["available_at"]
                output.append(
                    {
                        "security_id": sid,
                        "timestamp": start["timestamp"],
                        "available_at": start["available_at"],
                        "label_end_time": label_end,
                        "label_available_at": label_available,
                        "forward_return": label_value,
                    }
                )
        return pa.Table.from_pylist(output, schema=self.schema())

    def schema(self) -> pa.Schema:
        fields: list[pa.Field] = [
            pa.field("security_id", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("label_end_time", pa.timestamp("us", tz="UTC")),
            pa.field("label_available_at", pa.timestamp("us", tz="UTC")),
            pa.field("forward_return", pa.float64()),
        ]
        return pa.schema(
            fields,
            metadata={
                b"label_name": b"forward_close_return",
                b"label_version": b"1",
                b"horizon_observations": str(self.horizon).encode(),
                b"frequency": self.frequency.value.encode(),
                b"price_adjustment": b"raw",
                b"return_semantics": b"close_to_future_close_not_executable",
            },
        )
