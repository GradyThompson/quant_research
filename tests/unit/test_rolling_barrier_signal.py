"""Synthetic rolling stream checks. Never imports private model code."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.strategy.barrier_decision import BarrierAction, BarrierDecisionConfig
from quant_research.strategy.rolling_barrier_signal import stream_rolling_barrier_signals


class StubRollingModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.security_id = SecurityId("SEC-000001")

    def initialize(self, data: BarrierModelInput) -> None:
        self.calls.append("initialize")
        self.security_id = data.security_id

    def update(self, new_bars: pa.Table, *, decision_at: datetime) -> None:
        assert new_bars.num_rows == 1
        assert new_bars.column("available_at")[0].as_py() == decision_at
        self.calls.append("update")

    def forecast(
        self,
        *,
        decision_at: datetime,
        upper_barrier_pct: float,
        lower_barrier_pct: float,
        max_holding_minutes: int,
    ) -> BarrierForecast:
        self.calls.append("forecast")
        return BarrierForecast(self.security_id, decision_at, 0.65, 0.25, 0.10)


def _fixture() -> tuple[BarrierModelInput, pa.Table]:
    start = datetime(2025, 1, 10, 15, 0, tzinfo=UTC)
    sid = "SEC-000001"
    initial = BarrierModelInput(
        security_id=SecurityId(sid),
        decision_at=start,
        minute_bars=pa.table({"close": [100.0]}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=30,
    )
    bars = pa.table(
        {
            "security_id": [sid, sid],
            "timestamp": [start + timedelta(minutes=1), start + timedelta(minutes=2)],
            "available_at": [start + timedelta(minutes=2), start + timedelta(minutes=3)],
            "close": [100.1, 100.2],
        }
    )
    return initial, bars


def test_stream_incremental_calls_and_decisions() -> None:
    initial, bars = _fixture()
    model = StubRollingModel()
    results = list(
        stream_rolling_barrier_signals(
            model=model, initial=initial, new_bars=bars, config=BarrierDecisionConfig()
        )
    )
    assert model.calls == ["initialize", "update", "forecast", "update", "forecast"]
    assert [r.decision.action for r in results] == [BarrierAction.ENTER_LONG] * 2
    assert [r.forecast.generated_at for r in results] == bars.column("available_at").to_pylist()


def test_pending_entry_skips() -> None:
    initial, bars = _fixture()
    results = list(
        stream_rolling_barrier_signals(
            model=StubRollingModel(),
            initial=initial,
            new_bars=bars,
            config=BarrierDecisionConfig(),
            has_pending_entry=True,
        )
    )
    assert all(r.decision.action is BarrierAction.SKIP for r in results)


@pytest.mark.parametrize("case", ["duplicate", "future", "security", "reorder"])
def test_invalid_batch_rejected_before_initialization(case: str) -> None:
    initial, bars = _fixture()
    data = bars.to_pydict()
    if case == "duplicate":
        data["timestamp"][1] = data["timestamp"][0]
    elif case == "future":
        data["available_at"][0] = initial.decision_at
    elif case == "security":
        data["security_id"][0] = "SEC-000002"
    else:
        data["available_at"][1] = data["available_at"][0]
    model = StubRollingModel()
    with pytest.raises(ValueError):
        list(
            stream_rolling_barrier_signals(
                model=model,
                initial=initial,
                new_bars=pa.table(data),
                config=BarrierDecisionConfig(),
            )
        )
    assert model.calls == []
