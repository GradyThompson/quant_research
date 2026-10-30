"""Synthetic checks for the incremental barrier model interface."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.model.rolling_barrier import RollingBarrierModel


class FakeRollingModel:
    def __init__(self) -> None:
        self.security_id = SecurityId("SEC-000001")
        self.total_bars = 0

    def initialize(self, data: BarrierModelInput) -> None:
        self.security_id = data.security_id
        self.total_bars = data.minute_bars.num_rows

    def update(self, new_bars: pa.Table, *, decision_at: datetime) -> None:
        self.total_bars += new_bars.num_rows

    def forecast(
        self,
        *,
        decision_at: datetime,
        upper_barrier_pct: float,
        lower_barrier_pct: float,
        max_holding_minutes: int,
    ) -> BarrierForecast:
        return BarrierForecast(
            security_id=self.security_id,
            generated_at=decision_at,
            probability_upper_first=0.6,
            probability_lower_first=0.3,
            probability_neither=0.1,
        )


def test_incremental_model_contract() -> None:
    model: RollingBarrierModel = FakeRollingModel()
    decision_at = datetime(2025, 1, 10, 21, 1, tzinfo=UTC)
    history = BarrierModelInput(
        security_id=SecurityId("SEC-000001"),
        decision_at=decision_at,
        minute_bars=pa.table({"close": [1.0, 2.0]}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )
    model.initialize(history)
    model.update(pa.table({"close": [3.0]}), decision_at=decision_at)
    result = model.forecast(
        decision_at=decision_at,
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=60,
    )
    assert isinstance(model, FakeRollingModel)
    assert model.total_bars == 3
    assert result.security_id == history.security_id
    assert result.generated_at == decision_at
