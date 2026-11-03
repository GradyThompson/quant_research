"""P5-A.3 reference evidence collection uses pinned on-disk JSON records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_p5_reference_actions.py"
SPEC = importlib.util.spec_from_file_location("audit_p5_reference_actions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def _report() -> dict:
    return {
        "holdout_accessed": False,
        "results": {
            "10d_positive_short_all": {
                "4.0": {
                    "trades_detail": [
                        {
                            "formation_day": "2024-01-02",
                            "entry_day": "2024-01-03",
                            "exit_day": "2024-01-17",
                            "ticker": "ABC",
                            "factor": "SPY",
                        },
                        {
                            "formation_day": "2024-01-03",
                            "entry_day": "2024-01-04",
                            "exit_day": "2024-01-18",
                            "ticker": "ABC",
                            "factor": "SPY",
                        },
                    ],
                }
            }
        },
    }


def test_collect_is_cached_and_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = Mock(return_value={"results": [], "recorded_at": "2026-11-01"})
    monkeypatch.setattr(script, "_fetch_paged", provider)
    a = script.collect_evidence(
        report=_report(), api_key="local-test", cache_root=tmp_path, horizon=10, cost_bps=4.0
    )
    assert len(a["references"]) == 2
    assert set(a["splits"]) == {"ABC", "SPY"}
    assert a["split_start"] == "2024-01-03"
    assert a["split_end"] == "2024-01-18"
    assert provider.call_count == 4
    b = script.collect_evidence(
        report=_report(), api_key="local-test", cache_root=tmp_path, horizon=10, cost_bps=4.0
    )
    assert a["references"].keys() == b["references"].keys()
    assert provider.call_count == 4  # All four calls reused evidence caches.


def test_protected_holdout_stops_any_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = Mock()
    monkeypatch.setattr(script, "_fetch_paged", provider)
    report = _report()
    report["holdout_accessed"] = True
    with pytest.raises(ValueError, match="holdout"):
        script.collect_evidence(
            report=report, api_key="local-test", cache_root=tmp_path, horizon=10, cost_bps=4.0
        )
    provider.assert_not_called()


def test_cached_evidence_tamper_detected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = Mock(return_value={"results": []})
    monkeypatch.setattr(script, "_fetch_paged", provider)
    params = {"ticker": "ABC", "market": "stocks"}
    script._cached_get(
        cache_root=tmp_path, family="v3/reference/tickers", parameters=params, key="test"
    )
    file = next(tmp_path.rglob("*.json"))
    record = json.loads(file.read_text())
    record["query"] = "tampered"
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mismatch"):
        script._cached_get(
            cache_root=tmp_path, family="v3/reference/tickers", parameters=params, key="test"
        )
