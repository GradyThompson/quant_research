"""P4.5 historical driver: protected dates and genuine market-wide first-15 data."""

import gzip
import hashlib
import json
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research_cross_sectional_orb.py"
SPEC = spec_from_file_location("research_cross_sectional_orb_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _market_file(path, day, opening, *, high_volume=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("ticker,window_start,open,high,low,close,volume\n")
        for i in range(390):
            ns = int((opening + timedelta(minutes=i)).timestamp()) * 1_000_000_000
            for ticker in ("AAA", "BBB", "QQQ"):
                price = (
                    102
                    if ticker == "AAA" and i >= 16
                    else (101 if ticker == "QQQ" and i >= 16 else 100)
                )
                vol = 1000 * (3 if ticker == "AAA" and high_volume and i < 15 else 1)
                stream.write(f"{ticker},{ns},{price},{price + 0.1},{price - 0.1},{price},{vol}\n")
    return path


@pytest.mark.parametrize("summary_workers", [1, 2])
def test_market_driver_uses_only_allowed_dates_and_generates_scored_events(
    monkeypatch, tmp_path, summary_workers
):
    calendar = xcals.get_calendar("XNYS")
    days = [
        d.date()
        for d in calendar.sessions_in_range(pd.Timestamp("2024-06-10"), pd.Timestamp("2024-07-23"))
    ]
    assert len(days) > 20
    protected = days[2]
    touched = []
    files = {}
    for day in days:
        if day == protected:
            continue
        opening = calendar.session_open(pd.Timestamp(day)).to_pydatetime()
        path = _market_file(
            tmp_path / f"{day}.gz", day, opening, high_volume=day >= date(2024, 7, 15)
        )
        files[day] = path

    def verified(*, manifest_root, day):
        assert day != protected
        touched.append(day)
        path = files[day]
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(SCRIPT, "_verified_source", verified)
    monkeypatch.setattr(SCRIPT, "protected_research_days", lambda _: {protected})
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"original": "synthetic"}))
    report = SCRIPT.run_market_research(
        manifest_root=tmp_path,
        original_receipt_path=receipt,
        start=days[0],
        end=days[-1],
        validation_start=date(2024, 7, 1),
        output_dir=tmp_path / "report",
        horizons=(30,),
        costs_bps=(8.0,),
        lookback_sessions=5,
        bootstrap_replicates=10,
        progress_every=1000,
        summary_workers=summary_workers,
        summary_cache_root=tmp_path / "summary_cache",
        extract_cache_root=tmp_path / "extract_cache",
    )
    assert report["holdout_accessed"] is False
    assert protected.isoformat() in report["original_protected_days_skipped"]
    assert protected not in touched
    assert report["results"]["30"]["events"] >= 1
    scores = report["results"]["30"]["cohorts"]["high_rvol_all"]["cost_sensitivity"]
    assert scores[0]["trades"] >= 1
    assert (tmp_path / "report" / "validation.json").exists()


def test_parallel_summary_cache_matches_serial(tmp_path):
    calendar = xcals.get_calendar("XNYS")
    days = [date(2024, 6, 10), date(2024, 6, 11), date(2024, 6, 12)]
    jobs = []
    for day in days:
        stamp = pd.Timestamp(day)
        opening = calendar.session_open(stamp).to_pydatetime()
        closing = calendar.session_close(stamp).to_pydatetime()
        source = _market_file(tmp_path / f"source-{day}.gz", day, opening, high_volume=True)
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        jobs.append((day, source, sha, tmp_path / "cache", opening, closing))
    first = list(SCRIPT._ordered_summaries(jobs, workers=1))
    assert [cached for _, _, cached in first] == [False] * len(jobs)
    second = list(SCRIPT._ordered_summaries(jobs, workers=2))
    assert [day for day, _, _ in first] == [day for day, _, _ in second]
    assert [summaries for _, summaries, _ in first] == [summaries for _, summaries, _ in second]
    assert [cached for _, _, cached in second] == [True] * len(jobs)


def test_parallel_summary_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be positive"):
        list(SCRIPT._ordered_summaries([], workers=0))
