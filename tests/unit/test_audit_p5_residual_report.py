"""Unit checks for the read-only P5-A diagnostic report audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/audit_p5_residual_report.py"
SPEC = importlib.util.spec_from_file_location("audit_p5_residual_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _report() -> dict[str, Any]:
    return {
        "holdout_accessed": False,
        "configuration": {"max_per_side": 3, "eligible_days": 2},
        "results": {
            "10d_positive_short_all": {
                "4.0": {
                    "trades": 2,
                    "net_bps_per_entry_session": 34.0 / 6,
                    "block_bootstrap_95_interval": [-4.0, 12.0],
                    "trades_detail": [
                        {
                            "formation_day": "2024-01-02",
                            "entry_day": "2024-01-03",
                            "exit_day": "2024-01-17",
                            "ticker": "AAA",
                            "factor": "SPY",
                            "beta": 1.0,
                            "gross_factor_hedged_bps": 60.0,
                            "net_factor_hedged_bps": 52.0,
                            "unhedged_stock_bps": -64.0,
                        },
                        {
                            "formation_day": "2024-01-03",
                            "entry_day": "2024-01-04",
                            "exit_day": "2024-01-18",
                            "ticker": "BBB",
                            "factor": "QQQ",
                            "beta": 1.0,
                            "gross_factor_hedged_bps": -10.0,
                            "net_factor_hedged_bps": -18.0,
                            "unhedged_stock_bps": -20.0,
                        },
                    ],
                },
            },
        },
        "candidate_diagnostics": {"missing_or_discontinuous_exit": 3},
    }


def test_audit_rebuilds_returns_and_exposes_capital_overlap() -> None:
    audited = script.audit_report(_report())
    assert audited["trades"] == 2
    assert audited["reconstructed_net_bps_per_entry_session"] == pytest.approx(34 / 6)
    assert audited["net_bps_per_trade_mean"] == pytest.approx(17)
    assert audited["net_bps_per_trade_median"] == pytest.approx(17)
    assert audited["hedge_incremental_net_bps_per_trade_mean"] == pytest.approx(59)
    assert audited["best_trade_removal_net_bps_per_entry_session"]["remove_best_1"] == -3
    assert audited["open_positions_on_entry_dates"]["maximum"] == 2
    assert audited["top_tickers_by_trade_count"] == [("AAA", 1), ("BBB", 1)]
    assert audited["selection_diagnostics"]["missing_or_discontinuous_exit"] == 3


def test_audit_rejects_tampered_cost_or_report_average() -> None:
    report = _report()
    report["results"]["10d_positive_short_all"]["4.0"]["trades_detail"][0][
        "net_factor_hedged_bps"
    ] = 123.0
    with pytest.raises(ValueError, match="trade costs"):
        script.audit_report(report)
    report = _report()
    report["results"]["10d_positive_short_all"]["4.0"]["net_bps_per_entry_session"] = 99.0
    with pytest.raises(ValueError, match="does not match"):
        script.audit_report(report)


def test_audit_rejects_missing_holdout_guarantee_and_invalid_cost() -> None:
    report = _report()
    report["holdout_accessed"] = True
    with pytest.raises(ValueError, match="holdout"):
        script.audit_report(report)
    with pytest.raises(ValueError, match="available"):
        script.audit_report(_report(), cost_bps=12)


def test_audit_rejects_non_numeric_trade_value() -> None:
    report = _report()
    report["results"]["10d_positive_short_all"]["4.0"]["trades_detail"][0][
        "gross_factor_hedged_bps"
    ] = "60"
    with pytest.raises(ValueError, match="numeric"):
        script.audit_report(report)
