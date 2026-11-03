"""P4 synthetic fold/regression tests without market-data reads or broker access."""

from datetime import UTC, date, datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from math import exp, sin
from pathlib import Path

import pytest

from quant_research.model.first_passage_validation import MinuteBar, Session

# scripts/ is not part of the installed src-layout distribution. Load this
# entrypoint by its absolute repository path instead of requiring PYTHONPATH=.
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research_directional_returns_walkforward.py"
)
_spec = spec_from_file_location("p4_return_walkforward_cli_test", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
script = module_from_spec(_spec)
_spec.loader.exec_module(script)


def _make_sessions() -> tuple[list[Session], list[Session]]:
    assets, factors = [], []
    for day_index in range(8):
        day = date(2024, 12, 1) + timedelta(days=day_index)
        asset_bars, factor_bars = [], []
        for i in range(120):
            stamp = datetime(day.year, day.month, day.day, 14, 31, tzinfo=UTC) + timedelta(
                minutes=i
            )
            market = 100.0 * exp(0.00015 * i + 0.0002 * sin(0.32 * i + day_index))
            asset = 200.0 * exp(0.0002 * i + 0.0007 * sin(0.43 * i + day_index))
            available = stamp + timedelta(minutes=1)
            asset_bars.append(
                MinuteBar(stamp, available, asset, asset * 1.001, asset * 0.999, asset)
            )
            factor_bars.append(
                MinuteBar(stamp, available, market, market * 1.001, market * 0.999, market)
            )
        assets.append(Session(day, tuple(asset_bars)))
        factors.append(Session(day, tuple(factor_bars)))
    return assets, factors


def test_p4_writes_fold_report_without_loading_reserved_days(monkeypatch, tmp_path) -> None:
    asset, factor = _make_sessions()
    days = [session.day for session in asset]
    opened = []
    asset_entries = [
        {
            "session": str(day),
            "ticker": "AAPL",
            "dataset_id": str(day),
            "dataset_version": "asset-v1",
        }
        for day in days
    ]
    factor_entries = [
        {
            "session": str(day),
            "ticker": "QQQ",
            "dataset_id": str(day),
            "dataset_version": "factor-v1",
        }
        for day in days
    ]
    protected = ["2025-01-03", "2025-01-06"]

    monkeypatch.setattr(script, "read_pinned_receipt", lambda path: ({}, str(path)))
    monkeypatch.setattr(
        script,
        "select_receipt_window",
        lambda *args, **kwargs: (asset_entries, set(days[5:7]), protected),
    )
    monkeypatch.setattr(
        script,
        "select_pinned_factor_entries",
        lambda *args, **kwargs: factor_entries,
    )

    def fake_load(store, entries):
        opened.extend(entries)
        return asset if entries[0]["ticker"] == "AAPL" else factor

    monkeypatch.setattr(script, "load_sessions", fake_load)
    output = tmp_path / "run1"
    report = script.run_directional_return_walkforward(
        ticker="AAPL",
        factor_ticker="QQQ",
        asset_receipt_path=tmp_path / "asset",
        factor_receipt_path=tmp_path / "factor",
        store=object(),
        output_dir=output,
        evaluation_start=date(2025, 1, 13),
        horizons=[5, 10, 15],
        warmup_sessions=2,
        alpha_fit_sessions=2,
        fold_embargo_sessions=1,
        fold_validation_sessions=2,
        fold_step_sessions=2,
        min_training_rows=2,
        beta_lookback=70,
        min_beta_observations=60,
        bootstrap_replicates=10,
    )
    assert report["holdout_accessed"] is False
    assert report["original_protected_sessions_not_loaded"] == protected
    assert {entry["session"] for entry in opened}.isdisjoint(protected)
    for experiment in report["experiments"]:
        assert experiment["pooled_scores"]["scored"] > 0
        assert experiment["row_diagnostics"]["eligible_rows"] > 0
        assert experiment["folds"][0]["status"] == "scored"
        assert len(experiment["pooled_cost_aware_proxy"]) == 3
        assert experiment["pooled_cost_aware_proxy"][1]["roundtrip_cost_bps"] == 8.0
        assert experiment["pooled_cost_aware_proxy"][1]["proxy_not_executable"] is True
    assert (output / "validation.json").is_file()
    with pytest.raises(FileExistsError):
        script.run_directional_return_walkforward(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=tmp_path / "asset",
            factor_receipt_path=tmp_path / "factor",
            store=object(),
            output_dir=output,
            evaluation_start=date(2025, 1, 13),
            horizons=[5],
        )


def test_original_holdout_remains_protected_when_anchor_moves(monkeypatch, tmp_path) -> None:
    # This test fails before any market data is accessed.
    monkeypatch.setattr(script, "read_pinned_receipt", lambda path: ({}, "hash"))
    monkeypatch.setattr(
        script,
        "select_receipt_window",
        lambda *args, **kwargs: ([{"session": "2025-01-06"}], set(), ["2025-01-06"]),
    )
    monkeypatch.setattr(
        script, "load_sessions", lambda *args, **kwargs: pytest.fail("loaded protected data")
    )
    with pytest.raises(ValueError, match="original M2.5/P3"):
        script.run_directional_return_walkforward(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=tmp_path / "a",
            factor_receipt_path=tmp_path / "f",
            store=object(),
            output_dir=tmp_path / "run2",
            evaluation_start=date(2025, 2, 1),
            horizons=[5],
        )
