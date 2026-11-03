"""P5-A.3 accounting reconciliation unit tests, offline only."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from math import isclose, log

import pytest

from quant_research.model.p5_reference_reconciliation import (
    _share_multiplier,
    reconcile_p5_report,
)


def _fixture() -> tuple[dict, dict]:
    cost = 4.0
    beta = 1.0
    stock_log = log(0.75)
    factor_log = log(1.03)
    gross = -stock_log * 10000 + beta * factor_log * 10000
    net = gross - 2 * cost
    trade = {
        "ticker": "ABC",
        "factor": "SPY",
        "beta": beta,
        "formation_day": "2024-01-02",
        "entry_day": "2024-01-03",
        "exit_day": "2024-01-17",
        "gross_factor_hedged_bps": gross,
        "net_factor_hedged_bps": net,
        "unhedged_stock_bps": -stock_log * 10000 - cost,
    }
    report = {
        "holdout_accessed": False,
        "configuration": {"max_per_side": 3, "eligible_days": 1},
        "results": {
            "10d_positive_short_all": {
                "4.0": {
                    "trades": 1,
                    "trades_detail": [trade],
                    "net_bps_per_entry_session": net / 3,
                }
            }
        },
    }
    evidence = {
        "split_start": "2024-01-03",
        "split_end": "2024-01-17",
        "references": {
            "2024-01-02|ABC": {
                "results": [
                    {"ticker": "ABC", "type": "CS", "locale": "us", "share_class_figi": "BBG123"}
                ]
            }
        },
        "splits": {"ABC": {"results": []}, "SPY": {"results": []}},
    }
    return report, evidence


def test_real_25_percent_drop_not_assumed_a_split() -> None:
    report, evidence = _fixture()
    result = reconcile_p5_report(report, evidence)
    assert result["verified_subset_trades_crossing_splits"] == 0
    # -stock simple return 25%; factor long +3%, 8 bps of two-leg costs.
    assert isclose(
        result["verified_subset_arithmetic_bps_per_original_entry_session"], (2500 + 300 - 8) / 3
    )


def test_actual_4_for_3_split_removes_fictitious_price_drop() -> None:
    report, evidence = _fixture()
    evidence["splits"]["ABC"]["results"] = [
        {
            "execution_date": "2024-01-10",
            "split_from": 3,
            "split_to": 4,
            "adjustment_type": "forward_split",
        }
    ]
    result = reconcile_p5_report(report, evidence)
    assert result["verified_subset_trades_crossing_splits"] == 1
    # A 25% raw drop is entirely explained by the 4-for-3 share-count increase.
    assert isclose(
        result["verified_subset_arithmetic_bps_per_original_entry_session"], (300 - 8) / 3
    )


def test_etf_not_treated_as_common_stock() -> None:
    report, evidence = _fixture()
    evidence["references"]["2024-01-02|ABC"]["results"][0]["type"] = "ETF"
    result = reconcile_p5_report(report, evidence)
    assert result["verified_common_stock_trades_after_original_selection"] == 0
    assert result["excluded_or_unverified_counts"]["non_common_stock:ETF"] == 1


def test_missing_reference_fails_closed() -> None:
    report, evidence = _fixture()
    evidence["references"].clear()
    result = reconcile_p5_report(report, evidence)
    assert result["verified_common_stock_trades_after_original_selection"] == 0
    assert result["excluded_or_unverified_counts"]["missing_dated_reference"] == 1


def test_missing_split_evidence_fails_closed() -> None:
    report, evidence = _fixture()
    evidence["splits"].pop("SPY")
    result = reconcile_p5_report(report, evidence)
    assert result["excluded_or_unverified_counts"]["missing_split_evidence"] == 1


def test_exit_day_split_counted_entry_day_split_not_double_counted() -> None:
    rows = [
        {"execution_date": "2024-01-03", "split_from": 1, "split_to": 10},
        {"execution_date": "2024-01-17", "split_from": 3, "split_to": 4},
    ]
    multiplier, observed = _share_multiplier(rows, entry=date(2024, 1, 3), exit_=date(2024, 1, 17))
    assert isclose(multiplier, 4 / 3)
    assert len(observed) == 1


def test_inconsistent_cost_rejected() -> None:
    report, evidence = _fixture()
    broken = deepcopy(report)
    broken["results"]["10d_positive_short_all"]["4.0"]["trades_detail"][0][
        "net_factor_hedged_bps"
    ] -= 2
    with pytest.raises(ValueError, match="inconsistent original cost"):
        reconcile_p5_report(broken, evidence)


def test_missing_covering_split_window_fails_closed() -> None:
    report, evidence = _fixture()
    evidence["split_end"] = "2024-01-12"
    result = reconcile_p5_report(report, evidence)
    assert result["excluded_or_unverified_counts"]["split_window_not_covered"] == 1
