"""Focused tests for read-only P5-A raw-price discontinuity diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from math import log
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/audit_p5_split_risk.py"
SPEC = importlib.util.spec_from_file_location("audit_p5_split_risk", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _trade(*, stock_log_bps: float, day: str, ticker: str, beta: float = 1.0) -> dict[str, Any]:
    cost = 4.0
    short_stock_gross = -stock_log_bps
    factor_gross = 0.0
    gross = short_stock_gross + factor_gross
    net = gross - cost * (1 + beta)
    return {
        "formation_day": "2024-01-02",
        "entry_day": "2024-01-03",
        "exit_day": day,
        "ticker": ticker,
        "factor": "SPY",
        "beta": beta,
        "gross_factor_hedged_bps": gross,
        "net_factor_hedged_bps": net,
        "unhedged_stock_bps": short_stock_gross - cost,
    }


def _report() -> dict[str, Any]:
    split_like = _trade(
        stock_log_bps=10_000 * log(0.75),
        day="2024-01-18",
        ticker="AAA",
    )
    ordinary = _trade(stock_log_bps=-60, day="2024-01-19", ticker="BBB")
    trades = [split_like, ordinary]
    total = sum(row["net_factor_hedged_bps"] for row in trades)
    return {
        "holdout_accessed": False,
        "configuration": {"max_per_side": 1, "eligible_days": 2},
        "results": {
            "10d_positive_short_all": {
                "4.0": {
                    "trades": len(trades),
                    "trades_detail": trades,
                    "net_bps_per_entry_session": total / 2,
                }
            }
        },
        "candidate_diagnostics": {"missing_or_discontinuous_exit": 2},
    }


def test_detects_split_like_trade_and_preserves_denominator() -> None:
    result = script.audit_p5_split_risk(_report())
    assert result["split_like_ratio_trades"] == 1
    assert result["large_raw_decline_trades"] == 1
    assert result["top_trades"][0]["split_like_ratio"] == "4-for-3"
    assert result["top_trades"][0]["raw_open_to_open_price_change_pct"] == pytest.approx(-25)
    assert result["after_excluding_large_raw_declines_net_bps_per_entry_session"] == pytest.approx(
        26
    )
    assert result["selection_diagnostics"]["missing_or_discontinuous_exit"] == 2


def test_rejects_tampered_cost_and_score() -> None:
    report = _report()
    report["results"]["10d_positive_short_all"]["4.0"]["trades_detail"][0][
        "gross_factor_hedged_bps"
    ] += 1
    with pytest.raises(ValueError, match="transaction costs"):
        script.audit_p5_split_risk(report)
    report = _report()
    report["results"]["10d_positive_short_all"]["4.0"]["net_bps_per_entry_session"] = 0
    with pytest.raises(ValueError, match="reconcile"):
        script.audit_p5_split_risk(report)


def test_rejects_unprotected_report_and_bad_settings() -> None:
    report = _report()
    report["holdout_accessed"] = True
    with pytest.raises(ValueError, match="holdout"):
        script.audit_p5_split_risk(report)
    with pytest.raises(ValueError, match="cost scenario"):
        script.audit_p5_split_risk(_report(), cost_bps=8)
    with pytest.raises(ValueError, match="invalid diagnostic"):
        script.audit_p5_split_risk(_report(), large_drop_pct=-1)


def test_no_false_split_match_for_small_move() -> None:
    report = _report()
    record = report["results"]["10d_positive_short_all"]["4.0"]
    record["trades_detail"] = [record["trades_detail"][1]]
    record["trades"] = 1
    record["net_bps_per_entry_session"] = 26
    result = script.audit_p5_split_risk(report)
    assert result["split_like_ratio_trades"] == 0
    assert result["large_raw_decline_trades"] == 0
    assert result["after_excluding_large_raw_declines_net_bps_per_entry_session"] == pytest.approx(
        26
    )
