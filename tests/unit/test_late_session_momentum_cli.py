"""P4.6 script imports and protected-source access tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research_late_session_momentum.py"
SPEC = spec_from_file_location("research_late_session_momentum", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_protected_day_never_read_and_previous_close_not_bridged(tmp_path, monkeypatch) -> None:
    days = [
        date(2024, 3, 11),
        date(2024, 3, 12),
        date(2024, 3, 13),
        date(2024, 3, 14),
        date(2024, 3, 15),
    ]
    protected = days[1]
    receipt = tmp_path / "original_receipt.json"
    receipt.write_text(json.dumps({"test": "synthetic"}), encoding="utf-8")
    touched: list[date] = []

    def verified(*, manifest_root: Path, day: date) -> tuple[Path, str]:
        del manifest_root
        touched.append(day)
        return tmp_path / f"source-{day}.csv.gz", "synthetic_sha"

    def loaded(source, *, session, tickers, opening, closing, **kwargs):
        del source, closing, kwargs
        rows = {}
        for ticker in tickers:
            bars = []
            for i in range(390):
                end = opening + timedelta(minutes=i + 1)
                price = 100.0 + i / 1000.0
                bars.append(
                    {
                        "timestamp": end,
                        "available_at": end + timedelta(minutes=1),
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price,
                        "volume": 1000.0,
                    }
                )
            rows[ticker] = bars
        assert session != protected
        return rows, False

    monkeypatch.setattr(SCRIPT, "_verified_source", verified)
    monkeypatch.setattr(SCRIPT, "load_or_parse_selected_minutes", loaded)
    monkeypatch.setattr(SCRIPT, "protected_research_days", lambda _receipt: {protected})
    report = SCRIPT.run_late_session_research(
        manifest_root=tmp_path,
        original_receipt_path=receipt,
        start=days[0],
        end=days[-1],
        validation_start=days[2],
        output_dir=tmp_path / "report",
        tickers=("SPY",),
        bootstrap_replicates=20,
        progress_every=100,
    )
    assert protected not in touched
    assert report["holdout_accessed"] is False
    assert report["original_protected_days_skipped"] == [protected.isoformat()]
    assert report["results"]["SPY"]["scored_events"] == 2  # Mar 14 + 15
    assert (tmp_path / "report" / "validation.json").is_file()
    with pytest.raises(FileExistsError):
        SCRIPT.run_late_session_research(
            manifest_root=tmp_path,
            original_receipt_path=receipt,
            start=days[0],
            end=days[-1],
            validation_start=days[2],
            output_dir=tmp_path / "report",
            tickers=("SPY",),
        )


def test_manifest_hash_required(tmp_path) -> None:
    day = date(2024, 3, 11)
    source = tmp_path / "raw.csv.gz"
    source.write_bytes(b"payload")
    manifest = {
        "format": 1,
        "bucket": "flatfiles",
        "key": f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz",
        "revisions": [
            {
                "path": str(source),
                "sha256": hashlib.sha256(b"payload").hexdigest(),
                "bytes": len(b"payload"),
            }
        ],
    }
    (tmp_path / f"{day}.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert SCRIPT._verified_source(manifest_root=tmp_path, day=day)[0] == source
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        SCRIPT._verified_source(manifest_root=tmp_path, day=day)
