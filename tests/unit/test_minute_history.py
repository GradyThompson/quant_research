"""Offline tests for expanding a pinned historical minute-data receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from quant_research.data.ingestion.minute_history import expand_minute_history


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    plan_path = tmp_path / "old_plan.json"
    receipt_path = tmp_path / "old_receipt.json"
    new_plan = tmp_path / "new_plan.json"
    new_receipt = tmp_path / "new_receipt.json"
    plan = {
        "schema_version": 1,
        "status": "planned_only_no_data_downloaded",
        "initialization_sessions": ["2025-01-10"],
        "evaluation_session": "2025-01-13",
        "security_master_snapshot_version": "snapshot",
        "members": [{"ticker": "AAPL", "mic": "XNAS", "security_id": "SEC-1"}],
        "planned_requests": 2,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    entry = {
        "ticker": "AAPL",
        "mic": "XNAS",
        "security_id": "SEC-1",
        "session": "2025-01-10",
        "dataset_id": "market.ohlcv.minute.aapl.s20250110",
        "dataset_version": "00000000-0000-0000-0000-000000000001",
        "rows": 390,
    }
    receipt = {
        "schema_version": 1,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "snapshot_version": "snapshot",
        "publication_lag_seconds": 60,
        "entries": {"AAPL:XNAS:2025-01-10": entry},
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return plan_path, receipt_path, new_plan, new_receipt


def test_expand_reuses_exact_pinned_versions(tmp_path: Path) -> None:
    plan, receipt, output_plan, output_receipt = _setup(tmp_path)
    store = Mock()
    store.read.return_value.num_rows = 390
    store.read.return_value.column.return_value.to_pylist.return_value = ["SEC-1"]
    result = expand_minute_history(
        source_plan_path=plan,
        source_receipt_path=receipt,
        output_plan_path=output_plan,
        output_receipt_path=output_receipt,
        store=store,
        history_sessions=3,
    )
    assert result == {"reused": 1, "total": 4, "remaining": 3}
    expanded = json.loads(output_plan.read_text())
    assert len(expanded["initialization_sessions"]) == 3
    assert expanded["initialization_sessions"][-1] == "2025-01-10"
    new_receipt = json.loads(output_receipt.read_text())
    assert new_receipt["entries"]["AAPL:XNAS:2025-01-10"]["rows"] == 390
    assert new_receipt["plan_sha256"] == hashlib.sha256(output_plan.read_bytes()).hexdigest()
    assert store.read.call_count == 1


def test_expand_refuses_overwriting(tmp_path: Path) -> None:
    plan, receipt, output_plan, output_receipt = _setup(tmp_path)
    output_plan.write_text("existing")
    with pytest.raises(FileExistsError):
        expand_minute_history(
            source_plan_path=plan,
            source_receipt_path=receipt,
            output_plan_path=output_plan,
            output_receipt_path=output_receipt,
            store=Mock(),
            history_sessions=3,
        )


def test_expand_rejects_receipt_mismatch(tmp_path: Path) -> None:
    plan, receipt, output_plan, output_receipt = _setup(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["plan_sha256"] = "wrong"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        expand_minute_history(
            source_plan_path=plan,
            source_receipt_path=receipt,
            output_plan_path=output_plan,
            output_receipt_path=output_receipt,
            store=Mock(),
            history_sessions=3,
        )
