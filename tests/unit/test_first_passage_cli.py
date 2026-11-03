"""End-to-end holdout-read guard using pinned metadata and a fake store."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from runpy import run_path
from uuid import UUID

import pytest

pa = pytest.importorskip("pyarrow")

from quant_research.model.first_passage import BarrierScenario  # noqa: E402

# Pytest does not necessarily put the repository root on sys.path. Load the
# executable script by its filesystem path rather than importing "scripts".
run_validation = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "validate_first_passage.py")
)["run_validation"]


class RecordingStore:
    def __init__(self) -> None:
        self.read_ids: list[str] = []

    def read(self, dataset_id: object, version: object) -> object:
        self.read_ids.append(str(dataset_id))
        index = int(str(dataset_id).split("_")[-1])
        stamp = datetime.combine(
            date(2024, 10, 1) + timedelta(days=index), datetime.min.time(), tzinfo=UTC
        ) + timedelta(hours=15)
        return pa.table(
            {
                "security_id": ["SEC-1"],
                "timestamp": [stamp],
                "available_at": [stamp],
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
            }
        )


def test_validation_never_reads_holdout_and_refuses_overwrite(tmp_path: Path) -> None:
    receipt = {
        "schema_version": 1,
        "entries": {
            str(i): {
                "ticker": "AAPL",
                "session": (date(2024, 10, 1) + timedelta(days=i)).isoformat(),
                "security_id": "SEC-1",
                "dataset_id": f"test.session_{i}",
                "dataset_version": str(UUID(int=i + 1)),
                "rows": 1,
            }
            for i in range(60)
        },
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    store = RecordingStore()
    dest = tmp_path / "result"
    report = run_validation(
        ticker="AAPL",
        receipt_path=path,
        store=store,  # type: ignore[arg-type]
        output_dir=dest,
        evaluation_start=date(2025, 1, 13),
        scenarios=[BarrierScenario(0.02, 0.01, 60)],
        decision_stride=62,
        lookback_returns=10,
        min_returns=1,
    )
    assert len(store.read_ids) == 49
    assert all(f"test.session_{i}" not in store.read_ids for i in range(49, 60))
    assert report["holdout_accessed"] is False
    assert len(report["withheld_sessions_not_loaded"]) == 11
    assert (dest / "validation.json").exists()
    with pytest.raises(FileExistsError):
        run_validation(
            ticker="AAPL",
            receipt_path=path,
            store=store,  # type: ignore[arg-type]
            output_dir=dest,
            evaluation_start=date(2025, 1, 13),
            scenarios=[BarrierScenario(0.02, 0.01, 60)],
            decision_stride=62,
            lookback_returns=10,
            min_returns=1,
        )
