"""First full rolling strategy test uses only a synthetic prediction model."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa

from quant_research.backtesting.rolling_minute_backtest import run_rolling_minute_backtest
from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.strategy.barrier_decision import BarrierDecisionConfig
from quant_research.strategy.barrier_trade_plan import BarrierTradePlanConfig


class FakeModel:
    def initialize(self, data: BarrierModelInput) -> None:
        self.security_id = data.security_id
        self.updates = 0

    def update(self, new_bars: pa.Table, *, decision_at: datetime) -> None:
        self.updates += new_bars.num_rows

    def forecast(
        self,
        *,
        decision_at: datetime,
        upper_barrier_pct: float,
        lower_barrier_pct: float,
        max_holding_minutes: int,
    ) -> BarrierForecast:
        return BarrierForecast(self.security_id, decision_at, 0.8, 0.1, 0.1)


def test_complete_backtest_tracks_cash_and_blocks_overlaps() -> None:
    security = SecurityId("SEC-000001")
    start = datetime(2025, 1, 13, 14, 31, tzinfo=UTC)
    initial = BarrierModelInput(
        security_id=security,
        decision_at=start - timedelta(minutes=2),
        minute_bars=pa.table({"close": [100.0]}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=3,
    )
    ends = [start + timedelta(minutes=i) for i in range(12)]
    bars = pa.table(
        {
            "security_id": [str(security)] * len(ends),
            "timestamp": ends,
            "available_at": [t + timedelta(seconds=30) for t in ends],
            "open": [100.0] * len(ends),
            "high": [100.1] * len(ends),
            "low": [99.9] * len(ends),
            "close": [100.0] * len(ends),
        }
    )
    model = FakeModel()
    result = run_rolling_minute_backtest(
        model=model,
        initial=initial,
        bars=bars,
        session_close=start + timedelta(minutes=15),
        decision_config=BarrierDecisionConfig(),
        plan_config=BarrierTradePlanConfig(
            shares=10,
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=3,
        ),
    )
    assert result.forecasts == len(ends)
    assert model.updates == len(ends)
    assert len(result.trades) >= 1
    assert result.busy_signals > 0
    assert (
        abs(result.ending_cash - (result.initial_cash + sum(t.net_pnl for t in result.trades)))
        < 1e-7
    )
    assert all(
        left.exit.filled_at < right.entry.filled_at
        for left, right in zip(result.trades, result.trades[1:], strict=False)
    )


def test_incomplete_tail_is_not_booked() -> None:
    security = SecurityId("SEC-000001")
    start = datetime(2025, 1, 13, 14, 31, tzinfo=UTC)
    initial = BarrierModelInput(
        security_id=security,
        decision_at=start - timedelta(minutes=2),
        minute_bars=pa.table({"close": [100.0]}),
        upper_barrier_pct=0.005,
        lower_barrier_pct=0.004,
        max_holding_minutes=3,
    )
    ends = [start + timedelta(minutes=i) for i in range(3)]
    bars = pa.table(
        {
            "security_id": [str(security)] * len(ends),
            "timestamp": ends,
            "available_at": [t + timedelta(seconds=30) for t in ends],
            "open": [100.0] * len(ends),
            "high": [100.1] * len(ends),
            "low": [99.9] * len(ends),
            "close": [100.0] * len(ends),
        }
    )
    result = run_rolling_minute_backtest(
        model=FakeModel(),
        initial=initial,
        bars=bars,
        session_close=start + timedelta(minutes=15),
        decision_config=BarrierDecisionConfig(),
        plan_config=BarrierTradePlanConfig(
            shares=10,
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.004,
            max_holding_minutes=3,
        ),
    )
    assert not result.trades
    assert result.unavailable_entries > 0
    assert result.ending_cash == result.initial_cash
