"""Offline validation tests for minute pilot plan ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research.data.ingestion.minute_pilot_window import load_plan


def _plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "planned_only_no_data_downloaded",
        "initialization_sessions": ["2025-01-03", "2025-01-06"],
        "evaluation_session": "2025-01-13",
        "members": [{"security_id": "SEC-1", "ticker": "AAPL", "mic": "XNAS"}],
        "planned_requests": 3,
    }


def _save(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_plan_accepts_valid_sessions(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    _save(path, _plan())
    assert load_plan(path)["planned_requests"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("status", "complete"),
        ("planned_requests", 4),
        ("evaluation_session", "2025-01-11"),
        ("initialization_sessions", ["2025-01-06", "2025-01-03"]),
        ("members", [{"security_id": "SEC-1", "ticker": "AAPL.B", "mic": "XNAS"}]),
    ],
)
def test_load_plan_rejects_invalid_input(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "pilot.json"
    payload = _plan()
    payload[field] = value
    _save(path, payload)
    with pytest.raises(ValueError):
        load_plan(path)
