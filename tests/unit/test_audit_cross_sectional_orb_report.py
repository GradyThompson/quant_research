"""Standalone P4.5 report-audit tests with no historical data dependencies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit_cross_sectional_orb_report.py"
)
SPEC = importlib.util.spec_from_file_location("audit_cross_sectional_orb_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None

module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def trade(day: str, ticker: str, gross: float, side: int = 1) -> dict[str, object]:
    return {
        "day": day,
        "ticker": ticker,
        "side": side,
        "gross_bps": gross,
        "rvol": 2.0,
        "entry_at": f"{day}T10:00:00+00:00",
        "decision_at": f"{day}T09:59:00+00:00",
    }


def sample_report() -> dict[str, object]:
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    high = [trade(days[0], "AAA", 30.0), trade(days[1], "BBB", 14.0)]
    normal = [trade(days[0], "CCC", -2.0), trade(days[2], "DDD", -4.0)]
    confirmed = high[:1]
    cohorts = {}
    for name, trades in {
        "high_rvol_all": high,
        "high_rvol_market": confirmed,
        "normal_rvol_control_all": normal,
        "normal_rvol_control_market": normal[:1],
    }.items():
        scenarios = [
            {
                "roundtrip_cost_bps": 12.0,
                "trades": len(trades),
                "net_bps_per_session": sum((t["gross_bps"] - 12) / 2 for t in trades) / 4,
            }
        ]
        cohorts[name] = {"trades": trades, "cost_sensitivity": scenarios}
    return {
        "model_family": "cross-sectional-abnormal-volume-opening-range",
        "holdout_accessed": False,
        "configuration": {
            "max_positions": 2,
            "scored_sessions": 4,
            "validation_start": "2024-01-02",
        },
        "candidate_counts": {day: {} for day in days},
        "results": {"60": {"cohorts": cohorts}},
    }


def test_audit_reproduces_portfolio_and_paired_effect():
    audited = module.audit_report(sample_report(), cost=12.0, repeats=50)
    result = audited["horizons"]["60"]
    high = result["cohorts"]["high_rvol_all"]
    assert high["net_bps_per_session"] == pytest.approx(2.5)
    assert high["active_sessions"] == 2
    assert high["by_side"]["short"]["trades"] == 0
    assert high["after_removing_best_trades_bps_per_session"]["exclude_best_1"] == pytest.approx(
        0.25
    )
    assert result["comparisons"]["market_confirmation_vs_unfiltered"]["shared_trades"] == 1
    assert result["comparisons"]["market_confirmation_vs_unfiltered"]["second_only_trades"] == 1
    assert result["comparisons"]["high_vs_normal"]["shared_trades"] == 0


def test_audit_detects_mutated_published_scores():
    report = sample_report()
    report["results"]["60"]["cohorts"]["high_rvol_all"]["cost_sensitivity"][0][
        "net_bps_per_session"
    ] = 99
    with pytest.raises(ValueError, match="does not reproduce"):
        module.audit_report(report, repeats=10)


def test_audit_requires_known_protected_status():
    report = sample_report()
    report["holdout_accessed"] = True
    with pytest.raises(ValueError, match="holdout"):
        module.audit_report(report, repeats=10)


def test_audit_rejects_mismatched_session_count():
    report = sample_report()
    report["configuration"]["scored_sessions"] = 3
    with pytest.raises(ValueError, match="scored days"):
        module.audit_report(report, repeats=10)


def test_bootstrap_has_finite_ordered_limits():
    lo, hi = module._interval([0, 1, 2], repeats=100, seed=0)
    assert 0 <= lo <= hi <= 2


def test_cli_reports_missing_file_with_discovery_hint(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "not_the_real_smoke" / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_cross_sectional_orb_report.py",
            "--report",
            str(missing),
            "--output",
            str(tmp_path / "audit.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert str(missing) in stderr
    assert "find data/dev/models" in stderr
    assert not (tmp_path / "audit.json").exists()


def test_cli_accepts_report_directory(tmp_path, monkeypatch, capsys):
    import json

    report_dir = tmp_path / "actual_smoke"
    report_dir.mkdir()
    (report_dir / "validation.json").write_text(json.dumps(sample_report()), encoding="utf-8")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_cross_sectional_orb_report.py",
            "--report",
            str(report_dir),
            "--cost-bps",
            "12",
            "--bootstrap-replicates",
            "20",
            "--output",
            str(output),
        ],
    )
    module.main()
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["horizons"]["60"]["cohorts"]["high_rvol_all"]["trades"] == 2
    assert "Audit report:" in capsys.readouterr().out
