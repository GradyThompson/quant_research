"""Regression tests for P6 report-only diagnostic and independent trade arithmetic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_p6_ml_barrier_report.py"
SPEC = importlib.util.spec_from_file_location("audit_p6_ml_barrier_report_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _fixture() -> dict[str, object]:
    models = {}
    for name, amounts in {
        "math": [6.0, -2.0],
        "direct": [12.0, 4.0],
        "ensemble": [8.0, 2.0],
    }.items():
        dates = ("2024-03-01", "2024-03-04")
        trades = [
            {"day": date, "ticker": f"{name}-{i}", "net_bps": amount * 2}
            for i, (date, amount) in enumerate(zip(dates, amounts, strict=True))
        ]
        models[name] = {
            "scored": 100,
            "brier": 0.2,
            "log_loss": 0.8,
            "cost_scenarios": {
                "4.0": {
                    "trades_count": 2,
                    "trades": trades,
                    "daily_bps": dict(zip(dates, amounts, strict=True)),
                }
            },
        }
    return {
        "schema_version": 1,
        "model": "P6-ML-3outcome-cross-sectional-barrier",
        "configuration": {"positions": 2},
        "pooled": models,
    }


def test_report_audit_reconstructs_and_pairs() -> None:
    result = script.audit(_fixture())
    assert result["models"]["direct"]["slot_utilization"] == pytest.approx(0.5)
    assert result["models"]["direct"]["net_bps_per_session"] == pytest.approx(8)
    assert result["paired"]["direct_minus_math"][
        "mean_incremental_bps_per_session"
    ] == pytest.approx(6)
    assert result["paired"]["direct_minus_ensemble"][
        "mean_incremental_bps_per_session"
    ] == pytest.approx(3)
    assert result["models"]["direct"]["median_net_bps_per_trade"] == pytest.approx(16)


def test_audit_rejects_corrupt_trade_reconstruction() -> None:
    broken = _fixture()
    broken["pooled"]["direct"]["cost_scenarios"]["4.0"]["trades"][0]["net_bps"] = 999
    with pytest.raises(ValueError, match="reconstruct"):
        script.audit(broken)


def test_audit_rejects_misaligned_dates() -> None:
    broken = _fixture()
    broken["pooled"]["math"]["cost_scenarios"]["4.0"]["daily_bps"]["2024-03-05"] = 0
    with pytest.raises(ValueError, match="sessions differ"):
        script.audit(broken)


def test_block_ci_reproducible_and_rejects_empty() -> None:
    values = [0.1, 1.2, -1.7, 0.8, 2.3]
    assert script._block_ci(values, bootstrap_samples=30) == script._block_ci(
        values, bootstrap_samples=30
    )
    with pytest.raises(ValueError, match="no daily"):
        script._block_ci([])
