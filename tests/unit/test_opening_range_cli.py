"""Walk-forward CLI integration without actual OHLCV reads."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from test_opening_range import _sample

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research_opening_range.py"
_spec = spec_from_file_location("opening_range_script_test", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
script = module_from_spec(_spec)
_spec.loader.exec_module(script)


def test_pinned_staged_run_scores_and_never_opens_protected_days(monkeypatch, tmp_path) -> None:
    asset, factor = _sample(17)
    days = [day.day for day in asset]
    asset_entries = [
        {"session": str(day), "ticker": "AAPL", "dataset_id": str(day), "dataset_version": "v1"}
        for day in days
    ]
    factor_entries = [
        {"session": str(day), "ticker": "QQQ", "dataset_id": str(day), "dataset_version": "v1"}
        for day in days
    ]
    protected = ["2025-02-20", "2025-02-21"]
    opened = []
    monkeypatch.setattr(script, "read_pinned_receipt", lambda path: ({}, str(path)))
    monkeypatch.setattr(
        script,
        "select_receipt_window",
        lambda *args, **kwargs: (asset_entries, set(days[-2:]), protected),
    )
    monkeypatch.setattr(
        script, "select_pinned_factor_entries", lambda *args, **kwargs: factor_entries
    )

    def fake_load(store, entries):
        opened.extend(entries)
        return asset if entries[0]["ticker"] == "AAPL" else factor

    monkeypatch.setattr(script, "load_sessions", fake_load)
    output = tmp_path / "result"
    report = script.run_opening_range_research(
        ticker="AAPL",
        factor_ticker="QQQ",
        asset_receipt_path=tmp_path / "a",
        factor_receipt_path=tmp_path / "f",
        store=object(),
        output_dir=output,
        horizons=(30, 60),
        warmup_sessions=2,
        alpha_fit_sessions=2,
        fold_embargo_sessions=1,
        fold_validation_sessions=3,
        fold_step_sessions=3,
        bootstrap_replicates=10,
    )
    assert not report["holdout_accessed"]
    assert report["original_protected_sessions_not_loaded"] == protected
    assert {e["session"] for e in opened}.isdisjoint(protected)
    assert len(report["experiments"]) == 2
    for experiment in report["experiments"]:
        assert experiment["pooled_scored_days"] == 12
        assert len(experiment["folds"]) == 4
        scenarios = experiment["pooled_cost_sensitivity"]["market_volume_confirmed"]
        assert scenarios[1]["roundtrip_cost_bps"] == 8
        assert scenarios[1]["trades"] > 0
    assert (output / "validation.json").is_file()
    with pytest.raises(FileExistsError):
        script.run_opening_range_research(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=tmp_path / "a",
            factor_receipt_path=tmp_path / "f",
            store=object(),
            output_dir=output,
        )


def test_original_protected_guard_precedes_load(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(script, "read_pinned_receipt", lambda path: ({}, "hash"))
    monkeypatch.setattr(
        script,
        "select_receipt_window",
        lambda *args, **kwargs: ([{"session": "2025-01-06"}], set(), ["2025-01-06"]),
    )
    monkeypatch.setattr(
        script, "load_sessions", lambda *args, **kwargs: pytest.fail("protected data was read")
    )
    with pytest.raises(ValueError, match="original M2.5/P3"):
        script.run_opening_range_research(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=tmp_path / "a",
            factor_receipt_path=tmp_path / "f",
            store=object(),
            output_dir=tmp_path / "blocked",
        )
