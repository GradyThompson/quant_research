"""Offline verification tests with a synthetic pinned data store."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest

from quant_research.backtesting.execution import DailyBar
from quant_research.backtesting.verification import verify_baseline_report

T = datetime(2025, 1, 2, 21, tzinfo=UTC)
S = "SEC-000001"


def fixture():
    bars = tuple(
        DailyBar(S, T + timedelta(days=i), Decimal(p), Decimal(p), Decimal(p), Decimal(p), 1000)
        for i, p in enumerate(("100", "110", "120"))
    )
    availability = {(S, b.timestamp): b.timestamp + timedelta(minutes=15) for b in bars}
    config = {
        "store_root": "unused",
        "datasets": [
            {"dataset_id": "market.ohlcv.daily", "version": "550e8400-e29b-41d4-a716-446655440000"}
        ],
        "buy_security_id": S,
        "shares": 1,
        "initial_cash_usd": "1000",
    }
    commission = Decimal("0.05")
    price = Decimal("110.02")
    curve = [Decimal("1000"), Decimal("999.93"), Decimal("1009.93")]
    report = {
        "fills": [
            {
                "order_id": "buy",
                "security_id": S,
                "side": "buy",
                "shares": 1,
                "price": str(price),
                "commission": str(commission),
                "bar_timestamp": bars[1].timestamp.isoformat(),
            }
        ],
        "equity_curve": [
            {"timestamp": b.timestamp.isoformat(), "equity_usd": str(e)}
            for b, e in zip(bars, curve, strict=True)
        ],
        "initial_equity_usd": str(curve[0]),
        "final_equity_usd": str(curve[-1]),
        "total_return": str(curve[-1] / curve[0] - 1),
        "max_drawdown": str(curve[1] / curve[0] - 1),
    }
    return bars, availability, config, report


def test_valid_report():
    bars, availability, config, report = fixture()
    with patch(
        "quant_research.backtesting.verification.read_pinned_daily_data",
        return_value=(bars, availability),
    ):
        verify_baseline_report(config, report)


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("final_equity_usd", "1001", "final equity"),
        ("total_return", "0", "total return"),
        ("max_drawdown", "0", "max drawdown"),
    ],
)
def test_bad_report_rejected(key, value, reason):
    bars, availability, config, report = fixture()
    report[key] = value
    with patch(
        "quant_research.backtesting.verification.read_pinned_daily_data",
        return_value=(bars, availability),
    ):
        with pytest.raises(ValueError, match=reason):
            verify_baseline_report(config, report)


def test_same_bar_fill_rejected():
    bars, availability, config, report = fixture()
    report["fills"][0]["bar_timestamp"] = bars[0].timestamp.isoformat()
    with patch(
        "quant_research.backtesting.verification.read_pinned_daily_data",
        return_value=(bars, availability),
    ):
        with pytest.raises(ValueError, match="decision availability"):
            verify_baseline_report(config, report)


def test_equity_tampering_rejected():
    bars, availability, config, report = fixture()
    report["equity_curve"][1]["equity_usd"] = "1000"
    with patch(
        "quant_research.backtesting.verification.read_pinned_daily_data",
        return_value=(bars, availability),
    ):
        with pytest.raises(ValueError, match="equity curve"):
            verify_baseline_report(config, report)
