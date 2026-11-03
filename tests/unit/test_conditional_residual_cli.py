"""Source/holdout/overwrite checks for the new P5 experiment CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from quant_research.data.ingestion.minute_market_summary import DailySummary

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/research_conditional_residual.py"
SPEC = importlib.util.spec_from_file_location("research_conditional_residual", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def test_verified_source_rejects_unexpected_key_before_source_read(tmp_path: Path) -> None:
    day = date(2023, 1, 3)
    root = tmp_path / "manifests"
    root.mkdir()
    (root / f"{day}.json").write_text(
        json.dumps(
            {
                "format": 1,
                "bucket": "flatfiles",
                "key": "unexpected",
                "revisions": [{"path": str(tmp_path / "dont-read"), "sha256": "bad", "bytes": 5}],
            }
        )
    )
    with pytest.raises(ValueError, match="invalid pinned"):
        script._verified_source(manifest_root=root, day=day)


def test_run_excludes_protected_before_manifest_and_no_report_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = xcals.get_calendar("XNYS")
    days = [
        stamp.date()
        for stamp in calendar.sessions_in_range(
            pd.Timestamp("2023-01-03"), pd.Timestamp("2023-07-31")
        )
    ]
    protected = {days[50]}
    monkeypatch.setattr(script, "protected_research_days", lambda _: protected)
    visited: list[date] = []

    def fake_verify(*, manifest_root: Path, day: date) -> tuple[Path, str]:
        assert day not in protected, "protected source must not be verified"
        visited.append(day)
        return tmp_path / "unused", "0" * 64

    monkeypatch.setattr(script, "_verified_source", fake_verify)
    day_index = {day: i for i, day in enumerate(days)}

    def fake_summary(job: Any) -> tuple[date, dict[str, DailySummary], bool]:
        day = job[0]
        i = day_index[day]
        p = 100.0 + i * 0.12

        def daily(value: float) -> DailySummary:
            return DailySummary(
                opening_volume=100,
                opening_high=value + 0.1,
                opening_low=value - 0.1,
                opening_open=value,
                opening_close=value,
                last_close=value,
                dollar_volume=100_000_000.0,
            )

        return day, {"SPY": daily(p), "XLK": daily(p * 1.01), "AAA": daily(p)}, True

    monkeypatch.setattr(script, "summarize_market_job", fake_summary)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    destination = tmp_path / "report"
    report = script.run_research(
        start=days[0],
        end=days[-1],
        validation_start=days[70],
        manifest_root=tmp_path,
        original_receipt_path=receipt,
        output_dir=destination,
        factors=("SPY", "XLK"),
        estimation_sessions=20,
        holdings=(3,),
        max_universe=1,
        bootstrap_repeats=20,
        progress_every=500,
    )
    assert protected.isdisjoint(visited)
    assert report["holdout_accessed"] is False
    assert report["protected_dates_excluded_before_read"] == [days[50].isoformat()]
    assert report["total_scored_events"] == 0
    assert (destination / "validation.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        script.run_research(
            start=days[0],
            end=days[-1],
            validation_start=days[70],
            manifest_root=tmp_path,
            original_receipt_path=receipt,
            output_dir=destination,
        )
