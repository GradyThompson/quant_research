"""Synthetic end-to-end model comparison; source/chronology and report locks."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from quant_research.model.ml_barrier import FEATURE_NAMES, BarrierRow, row_to_json

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "research_ml_barrier.py"
SPEC = importlib.util.spec_from_file_location("research_ml_barrier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def make_row(day_index: int, ticker_index: int) -> BarrierRow:
    d = date(2024, 1, 1) + timedelta(days=day_index)
    first = datetime(d.year, d.month, d.day, 15, tzinfo=UTC)
    outcome = (day_index + ticker_index) % 3
    return BarrierRow(
        day=d,
        ticker=f"S{ticker_index}",
        decision_at=first,
        entry_at=first + timedelta(minutes=1),
        resolved_at=first + timedelta(hours=1),
        features=tuple(
            float((day_index % 5) + ticker_index + i) / 1000 for i in range(len(FEATURE_NAMES))
        ),
        baseline=(0.6, 0.1, 0.3),
        outcome=outcome,
        gross_return_bps=(50.0 if outcome == 0 else -200.0 if outcome == 1 else 15.0),
        timeout_return_bps=15.0 if outcome == 2 else 0,
        entry_price=100,
        ambiguous=False,
    )


def write_dataset(root: Path, *, include_protected: bool = False) -> None:
    root.mkdir()
    rows = [make_row(i, j) for i in range(100) for j in range(3)]
    if include_protected:
        rows.append(make_row(110, 0))
    path = root / "rows.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row_to_json(row)) + "\n")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protected_holdout_opened": False,
                "protected_dates": [(date(2024, 1, 1) + timedelta(days=110)).isoformat()],
                "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "counts": {"rows": len(rows)},
                "validation_start": "2024-03-15",
            }
        ),
        encoding="utf-8",
    )


def test_p6_full_report_and_fixed_cost_decisions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_dataset(root)
    report = script.run_research(
        dataset=root,
        output_dir=tmp_path / "report",
        min_train_sessions=40,
        validation_sessions=10,
        bootstrap_replicates=30,
        costs_bps=(4.0, 8.0),
    )
    assert report["holdout_accessed"] is False
    assert len(report["folds"]) >= 1
    for fold in report["folds"]:
        assert fold["train_last"] < fold["validation_first"]
    for model, value in report["pooled"].items():
        assert 0 <= value["brier"] <= 1, model
        assert (
            value["cost_scenarios"]["4.0"]["trades_count"]
            == value["cost_scenarios"]["8.0"]["trades_count"]
        )
    with pytest.raises(FileExistsError):
        script.run_research(dataset=root, output_dir=tmp_path / "report")


def test_protected_day_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    write_dataset(root, include_protected=True)
    with pytest.raises(ValueError, match="protected"):
        script.load_dataset(root)


def test_dataset_hash_protection(tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    write_dataset(root)
    with (root / "rows.jsonl.gz").open("ab") as out:
        out.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        script.load_dataset(root)


def test_early_validation_without_train_is_rejected() -> None:
    rows = [make_row(i, 0) for i in range(10)]
    with pytest.raises(ValueError, match="insufficient"):
        script.chronological_folds(rows, validation_start=date(2024, 1, 3))
