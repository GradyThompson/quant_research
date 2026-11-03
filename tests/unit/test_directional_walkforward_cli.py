"""Integration contract: walk-forward never accesses final holdout, or overwrites."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from runpy import run_path
from uuid import UUID

import pytest

pa = pytest.importorskip("pyarrow")

from quant_research.model.first_passage import BarrierScenario  # noqa: E402


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


def _receipt(ticker: str, prefix: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entries": {
            str(index): {
                "ticker": ticker,
                "session": (date(2024, 10, 1) + timedelta(days=index)).isoformat(),
                "security_id": "SEC-FACTOR" if ticker == "QQQ" else "SEC-ASSET",
                "dataset_id": f"{prefix}.session_{index}",
                "dataset_version": str(UUID(int=index + 1)),
                "rows": 90,
            }
            for index in range(20)
        },
    }


def test_walk_forward_pooled_no_holdout_access(tmp_path: Path) -> None:
    run = run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "research_directional_walkforward.py")
    )["run_directional_walkforward"]
    asset_receipt = tmp_path / "asset.json"
    factor_receipt = tmp_path / "factor.json"
    asset_receipt.write_text(json.dumps(_receipt("AAPL", "asset")))
    factor_receipt.write_text(json.dumps(_receipt("QQQ", "factor")))
    store = RecordingStore()
    destination = tmp_path / "walkforward"
    options = dict(
        ticker="AAPL",
        factor_ticker="QQQ",
        asset_receipt_path=asset_receipt,
        factor_receipt_path=factor_receipt,
        store=store,
        output_dir=destination,
        evaluation_start=date(2025, 1, 13),
        scenarios=[BarrierScenario(0.002, 0.002, 5)],
        training_sessions=6,
        validation_sessions=2,
        holdout_sessions=10,
        embargo_sessions=1,
        baseline_fit_sessions=2,
        alpha_fit_sessions=2,
        fold_embargo_sessions=1,
        fold_validation_sessions=2,
        fold_step_sessions=2,
        lookback_returns=30,
        min_returns=2,
        beta_lookback=15,
        min_beta_observations=5,
        min_directional_touches=3,
        bootstrap_replicates=10,
    )
    report = run(**options)
    assert report["fold_count"] == 2
    assert report["holdout_accessed"] is False
    assert len(report["withheld_sessions_not_loaded"]) == 11
    assert len(store.read_ids) == 18
    assert all(int(name.split("_")[-1]) < 9 for name in store.read_ids)
    assert (destination / "validation.json").exists()
    with pytest.raises(FileExistsError):
        run(**options)


def test_rejects_overlapping_validation_folds_before_io(tmp_path: Path) -> None:
    run = run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "research_directional_walkforward.py")
    )["run_directional_walkforward"]
    store = RecordingStore()
    with pytest.raises(ValueError, match="cannot overlap"):
        run(
            ticker="AAPL",
            factor_ticker="QQQ",
            asset_receipt_path=tmp_path / "none",
            factor_receipt_path=tmp_path / "none",
            store=store,
            output_dir=tmp_path / "output",
            evaluation_start=date(2025, 1, 13),
            scenarios=[BarrierScenario(0.002, 0.002, 5)],
            fold_step_sessions=1,
            fold_validation_sessions=2,
        )
    assert not store.read_ids
