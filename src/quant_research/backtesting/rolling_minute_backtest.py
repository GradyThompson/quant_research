"""First single-security, sequential long-only intraday research backtest."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

import pyarrow as pa

from quant_research.backtesting.minute_entry import (
    MinuteEntryConfig,
    simulate_next_minute_entry,
)
from quant_research.backtesting.minute_exit import MinuteExitConfig
from quant_research.backtesting.minute_portfolio import MinutePortfolioLedger
from quant_research.backtesting.minute_trade import MinuteTrade, simulate_minute_trade
from quant_research.model.barrier import BarrierModelInput
from quant_research.model.rolling_barrier import RollingBarrierModel
from quant_research.strategy.barrier_decision import BarrierAction, BarrierDecisionConfig
from quant_research.strategy.barrier_trade_plan import (
    BarrierTradePlanConfig,
    create_barrier_trade_plan,
)
from quant_research.strategy.rolling_barrier_signal import stream_rolling_barrier_signals


@dataclass(frozen=True, slots=True)
class RollingMinuteBacktestResult:
    initial_cash: float
    ending_cash: float
    forecasts: int
    approved_signals: int
    rejected_signals: int
    rejected_for_cash: int
    unavailable_entries: int
    busy_signals: int
    total_commissions: float
    trades: tuple[MinuteTrade, ...]

    @property
    def net_pnl(self) -> float:
        return self.ending_cash - self.initial_cash

    @property
    def wins(self) -> int:
        return sum(trade.net_pnl > 0 for trade in self.trades)


def run_rolling_minute_backtest(
    *,
    model: RollingBarrierModel,
    initial: BarrierModelInput,
    bars: pa.Table,
    session_close: datetime,
    decision_config: BarrierDecisionConfig,
    plan_config: BarrierTradePlanConfig,
    initial_cash: float = 10_000.0,
    entry_config: MinuteEntryConfig | None = None,
    exit_config: MinuteExitConfig | None = None,
) -> RollingMinuteBacktestResult:
    """Update model on every bar; execute at most one trade at a time.

    A completed trade is computed using future OHLC only *after* a qualifying
    signal. Its known exit timestamp blocks subsequent entries while the
    model continues receiving new bars. Entry signals always precede fill
    opens; exit fill costs and commissions are included in cash accounting.

    This is an offline research backtest, not a live order simulator.
    """
    if (
        plan_config.upper_barrier_pct != initial.upper_barrier_pct
        or plan_config.lower_barrier_pct != initial.lower_barrier_pct
        or plan_config.max_holding_minutes != initial.max_holding_minutes
    ):
        raise ValueError("Trade plan barriers/horizon must match model input")

    ledger = MinutePortfolioLedger(initial_cash=initial_cash)
    trades: list[MinuteTrade] = []
    counts: Counter[str] = Counter()
    occupied_until: datetime | None = None
    last_bar_end = bars.column("timestamp")[-1].as_py() if bars.num_rows else None
    for result in stream_rolling_barrier_signals(
        model=model,
        initial=initial,
        new_bars=bars,
        config=decision_config,
    ):
        counts["forecasts"] += 1
        signal_at = result.forecast.generated_at
        if occupied_until is not None and signal_at <= occupied_until:
            counts["busy"] += 1
            continue
        if result.decision.action is not BarrierAction.ENTER_LONG:
            counts["rejected"] += 1
            continue
        counts["approved"] += 1
        plan = create_barrier_trade_plan(
            forecast=result.forecast,
            decision=result.decision,
            config=plan_config,
        )
        if plan is None:
            raise AssertionError("Approved decision yielded no trade plan")
        # A trade near the end of a partial dataset may have an executable
        # entry but no complete forward horizon. Do not score censored data.
        # Missing minutes *inside* a nominally complete horizon still raise.
        prospective_entry = simulate_next_minute_entry(
            plan=plan,
            bars=bars,
            session_close=session_close,
            config=entry_config,
        )
        if prospective_entry is None:
            counts["unavailable"] += 1
            continue
        deadline = prospective_entry.filled_at + timedelta(
            minutes=prospective_entry.max_holding_minutes
        )
        if not isinstance(last_bar_end, datetime) or last_bar_end < deadline:
            counts["unavailable"] += 1
            continue
        trade = simulate_minute_trade(
            plan=plan,
            bars=bars,
            session_close=session_close,
            entry_config=entry_config,
            exit_config=exit_config,
        )
        if trade is None:
            counts["unavailable"] += 1
            continue
        required_cash = trade.entry.shares * trade.entry.fill_price + trade.entry.commission
        if required_cash > ledger.cash + 1e-9:
            counts["cash"] += 1
            continue
        ledger.book(trade)
        trades.append(trade)
        occupied_until = trade.exit.filled_at

    return RollingMinuteBacktestResult(
        initial_cash=initial_cash,
        ending_cash=ledger.cash,
        forecasts=counts["forecasts"],
        approved_signals=counts["approved"],
        rejected_signals=counts["rejected"],
        rejected_for_cash=counts["cash"],
        unavailable_entries=counts["unavailable"],
        busy_signals=counts["busy"],
        total_commissions=sum(t.entry.commission + t.exit.commission for t in trades),
        trades=tuple(trades),
    )
