"""Pinned factor/asset access and immutable research-output contract."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from runpy import run_path
from uuid import UUID

import pytest

pa = pytest.importorskip("pyarrow")

from quant_research.model.first_passage import BarrierScenario  # noqa: E402

run_research = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "research_directional.py")
)["run_directional_research"]


class RecordingStore:
    def __init__(self) -> None:
        self.read_ids: list[str] = []

    def read(self, dataset_id: object, version: object) -> object:
        self.read_ids.append(str(dataset_id))
        factor = str(dataset_id).startswith("factor.")
        index = int(str(dataset_id).split("_")[-1])
        day = date(2024, 10, 1) + timedelta(days=index)
        start = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=15)
        rows: list[dict[str, object]] = []
        previous = 100.0
        for minute in range(90):
            stamp = start + timedelta(minutes=minute + 1)
            change = (1 if minute % 2 else -1) * 0.0005 + (0.0001 if not factor else 0)
            close = previous * (1 + change)
            rows.append(
                {
                    "timestamp": stamp,
                    "available_at": stamp,
                    "security_id": "SEC-FACTOR" if factor else "SEC-ASSET",
                    "open": previous,
                    "high": max(previous, close),
                    "low": min(previous, close),
                    "close": close,
                }
            )
            previous = close
        return pa.Table.from_pylist(rows)


def receipt_entries(ticker: str, prefix: str) -> dict[str, object]:
    return {
        str(i): {
            "ticker": ticker,
            "session": (date(2024, 10, 1) + timedelta(days=i)).isoformat(),
            "security_id": "SEC-FACTOR" if ticker == "QQQ" else "SEC-ASSET",
            "dataset_id": f"{prefix}.session_{i}",
            "dataset_version": str(UUID(int=i + 1)),
            "rows": 90,
        }
        for i in range(20)
    }


def test_research_never_reads_holdout_and_does_not_overwrite(tmp_path: Path):
    asset_receipt = tmp_path / "asset_receipt.json"
    factor_receipt = tmp_path / "factor_receipt.json"
    asset_receipt.write_text(
        json.dumps({"schema_version": 1, "entries": receipt_entries("AAPL", "asset")})
    )
    factor_receipt.write_text(
        json.dumps({"schema_version": 1, "entries": receipt_entries("QQQ", "factor")})
    )
    store = RecordingStore()
    destination = tmp_path / "m3"
    kwargs = dict(
        ticker="AAPL",
        factor_ticker="QQQ",
        asset_receipt_path=asset_receipt,
        factor_receipt_path=factor_receipt,
        store=store,
        output_dir=destination,
        evaluation_start=date(2025, 1, 13),
        scenarios=[BarrierScenario(0.002, 0.002, 5)],
        training_sessions=6,
        seasonality_fit_sessions=2,
        validation_sessions=2,
        holdout_sessions=10,
        embargo_sessions=1,
        lookback_returns=30,
        min_returns=2,
        beta_lookback=15,
        min_beta_observations=5,
        min_directional_touches=3,
        bootstrap_replicates=20,
    )
    report = run_research(**kwargs)
    assert len(store.read_ids) == 18  # 6 train + 1 embargo + 2 validate, twice
    assert all(int(item.split("_")[-1]) < 9 for item in store.read_ids)
    assert report["holdout_accessed"] is False
    assert len(report["withheld_sessions_not_loaded"]) == 11
    assert (destination / "validation.json").exists()
    with pytest.raises(FileExistsError):
        run_research(**kwargs)


def test_missing_factor_sessions_fail_before_reading(tmp_path: Path):
    asset_receipt = tmp_path / "a.json"
    factor_receipt = tmp_path / "f.json"
    asset_receipt.write_text(
        json.dumps({"schema_version": 1, "entries": receipt_entries("AAPL", "asset")})
    )
    factor_receipt.write_text(json.dumps({"schema_version": 1, "entries": {}}))
    store = RecordingStore()
    with pytest.raises(ValueError, match="missing pinned QQQ"):
        run_research(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=asset_receipt,
            factor_receipt_path=factor_receipt,
            store=store,
            output_dir=tmp_path / "missing",
            evaluation_start=date(2025, 1, 13),
            scenarios=[BarrierScenario(0.002, 0.002, 5)],
            training_sessions=6,
            seasonality_fit_sessions=2,
            validation_sessions=2,
            holdout_sessions=10,
            embargo_sessions=1,
        )
    assert not store.read_ids
