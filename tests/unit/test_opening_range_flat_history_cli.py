"""Flat-file adapter tests: protected sessions must not be opened."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import pytest

from quant_research.model.opening_range_history import protected_research_days

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research_opening_range_flat_history.py"
)
_spec = spec_from_file_location("p4_4_flat_script_test", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
script = module_from_spec(_spec)
_spec.loader.exec_module(script)


def _original_receipt() -> dict[str, object]:
    # Select 60 timestamps prior to frozen original evaluation anchor.
    days = xcals.get_calendar("XNYS").sessions_in_range(
        pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-10")
    )
    days = list(days)[-60:]
    return {
        "schema_version": 1,
        "entries": {
            str(i): {
                "session": d.date().isoformat(),
                "security_id": "legacy-aapl-id",
                "ticker": "AAPL",
            }
            for i, d in enumerate(days)
        },
    }


def _minutes_for(day: date, ticker: str) -> list[dict[str, object]]:
    start = datetime(
        day.year,
        day.month,
        day.day,
        9,
        31,
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)
    out: list[dict[str, object]] = []
    for i in range(150):
        stamp = start + timedelta(minutes=i)
        close = (
            (100.0 if i < 18 else 100.3 + 0.04 * (i - 18)) if ticker == "AAPL" else 100 + 0.005 * i
        )
        out.append(
            {
                "timestamp": stamp,
                "available_at": stamp + timedelta(minutes=1),
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 500,
            }
        )
    return out


def test_p4_4_never_reads_original_protected_files(monkeypatch, tmp_path) -> None:
    original = _original_receipt()
    receipt_path = tmp_path / "original.json"
    receipt_path.write_text(json.dumps(original), encoding="utf-8")
    protected = protected_research_days(original)
    touched: list[date] = []

    def verified(*, manifest_root, day):
        assert day not in protected
        touched.append(day)
        return tmp_path / f"{day}.csv.gz", {"sha256": "synthetic"}

    def parser(path, *, session, tickers, **kwargs):
        assert session not in protected
        return {ticker: _minutes_for(session, ticker) for ticker in tickers}

    monkeypatch.setattr(script, "_verified_source", verified)
    monkeypatch.setattr(script, "parse_selected_minutes", parser)
    result = script.run_flat_history_research(
        tickers=("AAPL",),
        factor_ticker="QQQ",
        manifest_root=tmp_path,
        original_receipt_path=receipt_path,
        start=date(2024, 11, 1),
        end=date(2025, 1, 20),
        validation_start=date(2024, 11, 20),
        output_dir=tmp_path / "result",
        bootstrap_replicates=10,
    )
    assert result["holdout_accessed"] is False
    assert touched and set(touched).isdisjoint(protected)
    assert result["tickers"]["AAPL"]["validation_sessions"] > 2
    assert (tmp_path / "result" / "validation.json").is_file()


def test_manifest_immutable_hash_and_missing_refuse(tmp_path) -> None:
    day = date(2024, 11, 1)
    cache = tmp_path / "cache.gz"
    cache.write_bytes(b"synthetic header, not valid gzip")
    source = tmp_path / f"{day}.json"
    payload = {
        "format": 1,
        "bucket": "flatfiles",
        "key": f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz",
        "revisions": [
            {
                "path": str(cache),
                "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "bytes": cache.stat().st_size,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    path, _ = script._verified_source(manifest_root=tmp_path, day=day)
    assert path == cache
    cache.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash/size"):
        script._verified_source(manifest_root=tmp_path, day=day)
    with pytest.raises(FileNotFoundError, match="manifest"):
        script._verified_source(manifest_root=tmp_path, day=date(2024, 11, 4))


def test_extract_cache_reuses_verified_sessions_without_reparsing(monkeypatch, tmp_path) -> None:
    original = _original_receipt()
    receipt_path = tmp_path / "original.json"
    receipt_path.write_text(json.dumps(original), encoding="utf-8")
    calls: list[date] = []

    def verified(*, manifest_root, day):
        return tmp_path / f"{day}.csv.gz", {"sha256": "c" * 64}

    def parser(path, *, session, tickers, **kwargs):
        calls.append(session)
        return {ticker: _minutes_for(session, ticker) for ticker in tickers}

    monkeypatch.setattr(script, "_verified_source", verified)
    monkeypatch.setattr(script, "parse_selected_minutes", parser)
    params = {
        "tickers": ("AAPL",),
        "factor_ticker": "QQQ",
        "manifest_root": tmp_path,
        "original_receipt_path": receipt_path,
        "start": date(2024, 11, 1),
        "end": date(2025, 1, 20),
        "validation_start": date(2024, 11, 20),
        "extract_cache_root": tmp_path / "extractions",
        "bootstrap_replicates": 10,
    }
    first = script.run_flat_history_research(
        **params,
        output_dir=tmp_path / "first",
    )
    assert len(calls) == len(first["source_manifests"])
    assert first["extraction_cache_hits"] == 0

    def parser_should_not_run(*args, **kwargs):
        raise AssertionError("expected all verified sources to use cache")

    monkeypatch.setattr(script, "parse_selected_minutes", parser_should_not_run)
    second = script.run_flat_history_research(
        **params,
        output_dir=tmp_path / "second",
    )
    assert second["extraction_cache_hits"] == len(first["source_manifests"])
    assert first["tickers"] == second["tickers"]
