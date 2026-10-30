"""Independent, offline checks of the public buy-once backtest smoke test.

Not a general multi-strategy accounting auditor. Uses immutable dataset pins to
reconstruct expected single-trade equity and check temporal constraints.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from quant_research.backtesting.runner import DatasetPin, read_pinned_daily_data
from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage.parquet import ParquetDatasetStore


def verify_baseline_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    """Raise ValueError if the pinned buy-once report fails independent checks."""
    pins = [
        DatasetPin(DatasetId(d["dataset_id"]), DatasetVersion(UUID(d["version"])))
        for d in config["datasets"]
    ]
    bars, availability = read_pinned_daily_data(ParquetDatasetStore(config["store_root"]), pins)
    if not bars:
        raise ValueError("no bars")
    selected = config["buy_security_id"]
    shares = config["shares"]
    initial_cash = Decimal(str(config["initial_cash_usd"]))
    fills = report["fills"]
    if len(fills) != 1:
        raise ValueError("baseline must have exactly one fill")
    fill = fills[0]
    if fill["security_id"] != selected or fill["side"] != "buy" or fill["shares"] != shares:
        raise ValueError("unexpected baseline fill")
    fill_time = datetime.fromisoformat(fill["bar_timestamp"])
    first_decision_bar = next((b for b in bars if b.security_id == selected), None)
    if first_decision_bar is None:
        raise ValueError("selected security has no bars")
    first_decision_time = max(
        availability[(b.security_id, b.timestamp)]
        for b in bars
        if b.timestamp == first_decision_bar.timestamp
    )
    if fill_time <= first_decision_time:
        raise ValueError("fill occurred before or at decision availability")
    if not any(b.security_id == selected and b.timestamp == fill_time for b in bars):
        raise ValueError("fill not matched to a pinned price bar")
    price = Decimal(fill["price"])
    commission = Decimal(fill["commission"])
    if price <= 0 or commission < 0:
        raise ValueError("invalid fill economics")
    points = report["equity_curve"]
    times = [datetime.fromisoformat(p["timestamp"]) for p in points]
    equities = [Decimal(p["equity_usd"]) for p in points]
    if not points or times != sorted(set(times)):
        raise ValueError("invalid equity timestamps")
    if not all(x.is_finite() for x in equities):
        raise ValueError("nonfinite equity")
    if times != sorted({b.timestamp for b in bars}):
        raise ValueError("equity timestamps do not match pinned bars")
    # The baseline holds only a single share position, starting from cash.
    marks = {b.timestamp: b.close for b in bars if b.security_id == selected}
    previous_mark: Decimal | None = None
    expected_curve: list[Decimal] = []
    for t in times:
        if t in marks:
            previous_mark = marks[t]
        if t < fill_time:
            expected_curve.append(initial_cash)
        else:
            if previous_mark is None:
                raise ValueError("missing current security mark")
            expected_curve.append(
                initial_cash - shares * price - commission + shares * previous_mark
            )
    if equities != expected_curve:
        raise ValueError("equity curve does not reconcile with fill and pinned closes")
    if Decimal(report["initial_equity_usd"]) != equities[0]:
        raise ValueError("reported initial equity mismatch")
    if Decimal(report["final_equity_usd"]) != equities[-1]:
        raise ValueError("reported final equity mismatch")
    if equities[0] <= 0:
        raise ValueError("initial equity must be positive")
    expected_return = equities[-1] / equities[0] - 1
    if Decimal(report["total_return"]) != expected_return:
        raise ValueError("total return mismatch")
    peak = equities[0]
    drawdown = Decimal(0)
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            drawdown = min(drawdown, equity / peak - 1)
    if Decimal(report["max_drawdown"]) != drawdown:
        raise ValueError("max drawdown mismatch")
