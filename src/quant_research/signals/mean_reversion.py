"""Cross-sectional mean-reversion scores from trailing close-to-close returns.

Scores are hypotheses, not orders or evidence of trading profitability.
Daily signals are available no earlier than the input feature availability.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from quant_research.data.schemas import BarFrequency
from quant_research.features import TrailingReturnFeature


@dataclass(frozen=True, slots=True)
class MeanReversionSignal:
    """Score securities as negative trailing returns, then rank at each bar end.

    A minimum of two valid scores at one timestamp is required to rank.
    Equal scores receive the same average-rank percentile in [0, 1].
    Higher scores and percentiles indicate stronger mean-reversion hypotheses.
    """

    lookback: int = 5
    frequency: BarFrequency = BarFrequency.DAILY

    def __post_init__(self) -> None:
        TrailingReturnFeature(self.lookback, self.frequency)

    def compute(self, bars: pa.Table) -> pa.Table:
        features = TrailingReturnFeature(self.lookback, self.frequency).compute(bars)
        rows = features.to_pylist()
        by_time: dict[datetime, list[int]] = defaultdict(list)
        output: list[dict[str, str | datetime | float | None]] = []

        for idx, row in enumerate(rows):
            trailing = row["trailing_return"]
            score = -trailing if trailing is not None else None
            output.append(
                {
                    "security_id": row["security_id"],
                    "timestamp": row["timestamp"],
                    "available_at": row["available_at"],
                    "signal": score,
                    "cross_sectional_rank": None,
                }
            )
            if score is not None:
                by_time[row["timestamp"]].append(idx)

        for indices in by_time.values():
            count = len(indices)
            if count < 2:
                continue
            ordered = sorted(
                indices, key=lambda idx: (output[idx]["signal"], output[idx]["security_id"])
            )
            start = 0
            while start < count:
                end = start + 1
                while (
                    end < count
                    and output[ordered[end]]["signal"] == output[ordered[start]]["signal"]
                ):
                    end += 1
                percentile = ((start + end - 1) / 2) / (count - 1)
                for idx in ordered[start:end]:
                    output[idx]["cross_sectional_rank"] = percentile
                start = end

        return pa.Table.from_pylist(output, schema=self.schema())

    def schema(self) -> pa.Schema:
        fields: list[pa.Field] = [
            pa.field("security_id", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("signal", pa.float64()),
            pa.field("cross_sectional_rank", pa.float64()),
        ]
        return pa.schema(
            fields,
            metadata={
                b"signal_name": b"mean_reversion",
                b"signal_version": b"1",
                b"lookback_observations": str(self.lookback).encode(),
                b"frequency": self.frequency.value.encode(),
                b"price_adjustment": b"raw",
                b"rank_definition": b"average_tie_rank_0_to_1",
            },
        )
