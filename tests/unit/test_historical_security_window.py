"""Resumable, date-scoped inventory discovery tests (no network)."""

from datetime import date
from pathlib import Path

import pytest

from quant_research.security_master.historical_window import reconstruct_dated_window

DAYS = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]


def test_bounded_resume(tmp_path: Path) -> None:
    calls: list[date] = []

    def discover(day: date) -> tuple[str, dict[str, object]]:
        calls.append(day)
        return day.isoformat(), {"included": 5}

    kwargs = dict(
        dates=DAYS,
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path / "store",
        discover_and_save=discover,
    )
    first = reconstruct_dated_window(**kwargs, max_sessions=1)
    assert len(first["results"]) == 1
    second = reconstruct_dated_window(**kwargs, max_sessions=2)
    assert len(second["results"]) == 3
    reconstruct_dated_window(**kwargs, max_sessions=2)
    assert calls == DAYS


def test_failures_retry_without_losing_successes(tmp_path: Path) -> None:
    calls: list[date] = []

    def discover(day: date) -> tuple[str, dict[str, object]]:
        calls.append(day)
        if day == DAYS[0] and calls.count(day) == 1:
            raise RuntimeError("temporary")
        return day.isoformat(), {}

    kwargs = dict(
        dates=DAYS[:2],
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path / "store",
        discover_and_save=discover,
    )
    first = reconstruct_dated_window(**kwargs, max_sessions=2)
    assert first["results"][DAYS[0].isoformat()]["status"] == "failed"
    second = reconstruct_dated_window(**kwargs, max_sessions=2)
    assert all(row["status"] == "ok" for row in second["results"].values())
    assert calls == [DAYS[0], DAYS[1], DAYS[0]]


def test_reject_mismatched_manifest(tmp_path: Path) -> None:
    kwargs = dict(
        manifest_path=tmp_path / "manifest.json",
        store_root=tmp_path / "store",
        discover_and_save=lambda day: (day.isoformat(), {}),
        max_sessions=1,
    )
    reconstruct_dated_window(dates=DAYS, **kwargs)
    with pytest.raises(ValueError, match="different dates"):
        reconstruct_dated_window(dates=DAYS[:1], **kwargs)
    with pytest.raises(ValueError, match="positive"):
        reconstruct_dated_window(
            dates=DAYS, max_sessions=0, **{k: v for k, v in kwargs.items() if k != "max_sessions"}
        )
