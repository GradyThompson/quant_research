"""Regression: bounded multiprocessing agrees with the serial P6 builder."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import exchange_calendars as xcals
import pandas as pd
import pytest


def _load_builder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    import quant_research

    root = Path(quant_research.__file__).resolve().parent
    name = "quant_research.data.ingestion"
    try:
        __import__(name)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        fake = ModuleType(name)
        fake.__path__ = [str(root / "data" / "ingestion")]
        monkeypatch.setitem(sys.modules, name, fake)
    # forkserver/spawn must be able to import the submitted worker functions by
    # module name. A dynamically named spec_from_file_location module cannot
    # be re-imported in a fresh worker interpreter.
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_root))
    return importlib.import_module("build_p6_barrier_dataset")


def _market_session(path: Path, day: date, opened: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("ticker,window_start,open,high,low,close,volume\n")
        for symbol in ("SPY", "FOO"):
            for minute in range(390):
                start = opened + pd.Timedelta(minutes=minute)
                ns = int(start.timestamp() * 1_000_000_000)
                price = 100 + (0.0001 if symbol == "FOO" else 0.0002) * minute
                handle.write(
                    f"{symbol},{ns},{price:.5f},{price + 0.001:.5f},"
                    f"{price - 0.001:.5f},{price:.5f},2000\n"
                )


def _read_dataset(path: Path) -> list[dict[str, object]]:
    with gzip.open(path / "rows.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_parallel_and_serial_build_identical_rows_and_reuse_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_builder(monkeypatch)
    cal = xcals.get_calendar("XNYS")
    days = [
        s.date()
        for s in cal.sessions_in_range(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-09"))
    ]
    assert len(days) >= 26
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    for day in days:
        stamp = pd.Timestamp(day)
        source = tmp_path / f"{day}.csv.gz"
        _market_session(source, day, cal.session_open(stamp))
        contents = source.read_bytes()
        (manifest_root / f"{day}.json").write_text(
            json.dumps(
                {
                    "format": 1,
                    "bucket": "flatfiles",
                    "key": f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz",
                    "revisions": [
                        {
                            "path": str(source),
                            "sha256": hashlib.sha256(contents).hexdigest(),
                            "bytes": len(contents),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(module, "protected_research_days", lambda _receipt: set())
    monkeypatch.setattr(module.MassiveEvidence, "split_days", lambda *_args: set())
    calls: list[date] = []

    def common_stocks(_self: object, day: date) -> set[str]:
        calls.append(day)
        return {"FOO"}

    monkeypatch.setattr(module.MassiveEvidence, "common_stocks", common_stocks)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    config = dict(
        start=days[0],
        end=days[-1],
        validation_start=days[21],
        receipt=receipt,
        manifest_root=manifest_root,
        summary_root=tmp_path / "summaries",
        extract_root=tmp_path / "extracts",
        evidence_root=tmp_path / "evidence",
        api_key="",
        universe=5,
        progress_every=100,
        day_cache_root=tmp_path / "day_cache",
    )
    parallel = module.build_dataset(**config, workers=2, output_dir=tmp_path / "parallel")
    assert parallel["counts"]["rows"] > 0
    baseline = module.build_dataset(
        **{
            **config,
            "day_cache_root": tmp_path / "serial_day_cache",
            "extract_root": tmp_path / "serial_extract",
        },
        workers=1,
        output_dir=tmp_path / "serial",
    )
    before = len(calls)
    reused = module.build_dataset(**config, workers=2, output_dir=tmp_path / "reuse")
    assert parallel["counts"] == baseline["counts"]
    assert parallel["source_revisions"] == baseline["source_revisions"]
    assert _read_dataset(tmp_path / "parallel") == _read_dataset(tmp_path / "serial")
    assert reused["counts"] == parallel["counts"]
    assert len(calls) == before  # completed daily caches, no extra API requests
    with pytest.raises(ValueError, match="invalid dataset"):
        module.build_dataset(**config, workers=0, output_dir=tmp_path / "invalid")


def test_parallel_preparation_never_submits_protected_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protected date itself has no source manifest; a worker must not open it."""
    module = _load_builder(monkeypatch)
    cal = xcals.get_calendar("XNYS")
    days = [
        s.date()
        for s in cal.sessions_in_range(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-09"))
    ]
    protected = days[5]
    monkeypatch.setattr(module, "protected_research_days", lambda _receipt: {protected})
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module.MassiveEvidence, "split_days", lambda *_args: set())
    monkeypatch.setattr(module.MassiveEvidence, "common_stocks", lambda *_args: {"FOO"})
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    for day in days:
        if day == protected:
            continue
        source = tmp_path / f"{day}.csv.gz"
        _market_session(source, day, cal.session_open(pd.Timestamp(day)))
        contents = source.read_bytes()
        (manifest_root / f"{day}.json").write_text(
            json.dumps(
                {
                    "format": 1,
                    "bucket": "flatfiles",
                    "key": f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz",
                    "revisions": [
                        {
                            "path": str(source),
                            "sha256": hashlib.sha256(contents).hexdigest(),
                            "bytes": len(contents),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    result = module.build_dataset(
        start=days[0],
        end=days[-1],
        validation_start=days[22],
        receipt=receipt,
        manifest_root=manifest_root,
        output_dir=tmp_path / "safe",
        evidence_root=tmp_path / "evidence",
        summary_root=tmp_path / "summaries",
        extract_root=tmp_path / "extracts",
        day_cache_root=tmp_path / "cache",
        api_key="",
        universe=2,
        workers=2,
        progress_every=100,
    )
    assert result["protected_holdout_opened"] is False
    assert all(source["day"] != str(protected) for source in result["source_revisions"])
