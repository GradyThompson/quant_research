"""Offline tests for bounded and resumable daily import scheduling."""

import json
from datetime import date
from pathlib import Path

import pytest

from quant_research.data.ingestion.massive_daily_window import process_daily_window


def test_bounded_resume_and_provenance(tmp_path: Path) -> None:
    security = tmp_path / "security.json"
    imports = tmp_path / "imports"
    imports.mkdir()
    days = ["2025-01-08", "2025-01-10"]
    security.write_text(
        json.dumps(
            {
                "dates": days,
                "results": {
                    d: {"status": "ok", "snapshot_version": f"snapshot-{i}"}
                    for i, d in enumerate(days)
                },
            }
        )
    )
    calls: list[date] = []

    def run(day: date, snapshot: str) -> dict[str, object]:
        calls.append(day)
        result: dict[str, object] = {
            "session": day.isoformat(),
            "snapshot_version": snapshot,
            "dataset_id": "market.day",
            "dataset_version": "version",
            "sha256": "checksum",
        }
        (imports / f"{day.isoformat()}.json").write_text(json.dumps(result))
        return result

    one = process_daily_window(
        security_manifest=security, import_root=imports, max_sessions=1, process_session=run
    )
    assert one == {"total": 2, "completed": 1, "attempted": 1, "failed_this_run": 0, "remaining": 1}
    two = process_daily_window(
        security_manifest=security, import_root=imports, max_sessions=1, process_session=run
    )
    assert two["completed"] == 2
    assert len(calls) == 2
    three = process_daily_window(
        security_manifest=security, import_root=imports, max_sessions=1, process_session=run
    )
    assert three["attempted"] == 0


def test_reject_conflicting_import_pin(tmp_path: Path) -> None:
    security = tmp_path / "security.json"
    security.write_text(
        json.dumps(
            {
                "dates": ["2025-01-10"],
                "results": {"2025-01-10": {"status": "ok", "snapshot_version": "expected"}},
            }
        )
    )
    imports = tmp_path / "imports"
    imports.mkdir()
    (imports / "2025-01-10.json").write_text(
        json.dumps(
            {
                "session": "2025-01-10",
                "snapshot_version": "different",
                "dataset_version": "v",
                "sha256": "x",
            }
        )
    )
    with pytest.raises(ValueError, match="Conflicting"):
        process_daily_window(
            security_manifest=security,
            import_root=imports,
            max_sessions=1,
            process_session=lambda *_: {},
        )
