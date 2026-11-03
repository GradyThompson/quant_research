"""ETF identity checks and read-free M3 factor planning."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

functions = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "ingest_research_factor.py")
)
lookup_etf = functions["lookup_etf"]
run_factor_ingestion = functions["run_factor_ingestion"]


class FakeReferenceClient:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.calls: list[dict[str, object]] = []

    def list_tickers(self, **kwargs: Any) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return self.records


def test_etf_lookup_requires_exact_dated_identity() -> None:
    row = {
        "ticker": "QQQ",
        "primary_exchange": "XNAS",
        "locale": "us",
        "type": "ETF",
        "share_class_figi": "BBG-QQQ-TEST",
    }
    client = FakeReferenceClient([row])
    identity = lookup_etf(client, day=date(2025, 1, 10), ticker="QQQ", mic="XNAS")
    assert identity.ticker == "QQQ"
    assert identity.mic == "XNAS"
    assert identity.figi == "BBG-QQQ-TEST"
    assert identity.security_id.startswith("SEC-")
    assert client.calls == [
        {
            "ticker": "QQQ",
            "market": "stocks",
            "type": "ETF",
            "date": "2025-01-10",
            "active": True,
            "limit": 100,
        }
    ]
    with pytest.raises(ValueError, match="got 0"):
        lookup_etf(
            FakeReferenceClient([{**row, "type": "CS"}]),
            day=date(2025, 1, 10),
            ticker="QQQ",
            mic="XNAS",
        )
    with pytest.raises(ValueError, match="got 2"):
        lookup_etf(FakeReferenceClient([row, row]), day=date(2025, 1, 10), ticker="QQQ", mic="XNAS")
    with pytest.raises(ValueError, match="share-class FIGI"):
        lookup_etf(
            FakeReferenceClient([{**row, "share_class_figi": None}]),
            day=date(2025, 1, 10),
            ticker="QQQ",
            mic="XNAS",
        )


def test_dry_run_selects_no_holdout_and_no_data_reads(tmp_path: Path) -> None:
    first = date(2024, 10, 1)
    days = [first + timedelta(days=i) for i in range(60)]
    entries = {
        str(i): {
            "session": day.isoformat(),
            "ticker": "AAPL",
            "security_id": "SEC-AAPL",
            "dataset_id": f"asset.session_{i}",
            "dataset_version": "unused",
            "rows": 390,
        }
        for i, day in enumerate(days)
    }
    receipt = tmp_path / "asset.json"
    receipt.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    factor_receipt = tmp_path / "factor.json"

    class NoReads:
        def read(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("No data may be read in dry-run")

    result = run_factor_ingestion(
        asset_receipt_path=receipt,
        factor_receipt_path=factor_receipt,
        store=NoReads(),
        ticker="AAPL",
        factor_ticker="QQQ",
        factor_mic="XNAS",
        evaluation_start=date(2025, 1, 13),
        max_sessions=5,
        publication_lag_seconds=60,
        dry_run=True,
    )
    assert result["missing"] == 49
    assert result["remaining_days"] == [day.isoformat() for day in days[:49]]
    assert all(day.isoformat() not in result["remaining_days"] for day in days[49:])
    assert not factor_receipt.exists()
