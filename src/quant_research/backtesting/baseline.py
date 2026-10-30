"""Run an offline, pinned US-equity daily backtest with a PUBLIC test policy.

No broker, data provider, or private strategy code is contacted.
The policy buys a fixed share count once, subject to the engine's V1 limits.
This is an integration smoke test, not a validated investment strategy.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from quant_research.backtesting.engine import BacktestContext
from quant_research.backtesting.execution import HistoricalOrder, OrderType, Side
from quant_research.backtesting.runner import DatasetPin, run_pinned_backtest
from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.storage.parquet import ParquetDatasetStore
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId
from quant_research.trading.ledger import PortfolioLedger


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    if not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise ValueError("datasets must be a nonempty list")
    if not isinstance(data.get("instruments"), list) or not data["instruments"]:
        raise ValueError("instruments must be a nonempty list")
    if type(data.get("shares")) is not int or data["shares"] <= 0:
        raise ValueError("shares must be a positive integer")
    cash = Decimal(str(data.get("initial_cash_usd")))
    if not cash.is_finite() or cash <= 0:
        raise ValueError("initial_cash_usd must be positive and finite")
    return data


def run(config: dict[str, Any]) -> dict[str, Any]:
    store = ParquetDatasetStore(config["store_root"])
    pins = [
        DatasetPin(DatasetId(entry["dataset_id"]), DatasetVersion(entry["version"]))
        for entry in config["datasets"]
    ]
    instruments: dict[str, Instrument] = {}
    for entry in config["instruments"]:
        sid = SecurityId(entry["security_id"])
        if sid.value in instruments:
            raise ValueError(f"duplicate instrument {sid.value}")
        instruments[sid.value] = Instrument(sid, Market.US, "USD")
    selected = config["buy_security_id"]
    if selected not in instruments:
        raise ValueError("buy_security_id must appear in instruments")
    shares = config["shares"]
    ledger = PortfolioLedger(
        portfolio_id=PortfolioId("baseline_backtest"),
        mode=Mode.AUTOMATED_ALPHA,
        cash={Currency.USD: Decimal(str(config["initial_cash_usd"]))},
    )
    submitted = False

    def baseline_policy(context: BacktestContext) -> tuple[HistoricalOrder, ...]:
        nonlocal submitted
        if submitted or selected not in context.bars:
            return ()
        submitted = True
        return (
            HistoricalOrder(
                order_id="baseline-buy-001",
                security_id=selected,
                side=Side.BUY,
                kind=OrderType.MARKET,
                quantity=shares,
                submitted_at=context.decision_at,
            ),
        )

    result = run_pinned_backtest(
        store=store,
        pins=pins,
        instruments=instruments,
        initial_ledger=ledger,
        policy=baseline_policy,
    )
    return {
        "warning": "Smoke test only: raw prices, no corporate actions, borrow, or portfolio risk gate.",
        "start": result.equity_curve[0].timestamp.isoformat(),
        "end": result.equity_curve[-1].timestamp.isoformat(),
        "observations": len(result.equity_curve),
        "initial_equity_usd": str(result.equity_curve[0].equity_usd),
        "final_equity_usd": str(result.equity_curve[-1].equity_usd),
        "total_return": str(result.total_return) if result.total_return is not None else None,
        "max_drawdown": str(result.max_drawdown) if result.max_drawdown is not None else None,
        "fills": [
            {
                "order_id": fill.order_id,
                "security_id": fill.security_id,
                "side": fill.side.value,
                "shares": fill.quantity,
                "price": str(fill.price),
                "commission": str(fill.commission),
                "bar_timestamp": fill.bar_timestamp.isoformat(),
            }
            for fill in result.fills
        ],
        "unfilled_order_ids": [o.order_id for o in result.open_orders],
        "equity_curve": [
            {"timestamp": p.timestamp.isoformat(), "equity_usd": str(p.equity_usd)}
            for p in result.equity_curve
        ],
    }
