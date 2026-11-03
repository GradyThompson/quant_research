"""Test dataset version discovery without any private model or Massive network."""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from quant_research.backtesting.minute_run_setup import (
    resolve_initial_decision_at,
    resolve_update_dataset,
)
from quant_research.core import DatasetId, DatasetVersion


def _receipt(tmp_path: Path) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "sessions": [
                    {"session": "2025-01-10"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_initial_time(tmp_path: Path) -> None:
    assert resolve_initial_decision_at(_receipt(tmp_path)) == datetime(
        2025,
        1,
        10,
        21,
        1,
        tzinfo=UTC,
    )


def test_auto_selects_latest_session_version(tmp_path: Path) -> None:
    version1, version2 = DatasetVersion(uuid4()), DatasetVersion(uuid4())
    store = Mock()
    store.list_versions.return_value = [Mock(version=version1), Mock(version=version2)]
    dataset_id, version = resolve_update_dataset(
        store=store, receipt=_receipt(tmp_path), session=date(2025, 1, 13)
    )
    assert dataset_id == DatasetId("market.ohlcv.minute.aapl.s20250113")
    assert version == version2
    store.list_versions.assert_called_once_with(dataset_id)


def test_explains_missing_dataset(tmp_path: Path) -> None:
    store = Mock()
    store.list_versions.return_value = []
    with pytest.raises(ValueError, match="No stored minute dataset"):
        resolve_update_dataset(
            store=store,
            receipt=_receipt(tmp_path),
            session=date(2025, 1, 13),
        )


def test_explicit_version_must_exist(tmp_path: Path) -> None:
    store = Mock()
    store.list_versions.return_value = [Mock(version=DatasetVersion(uuid4()))]
    with pytest.raises(ValueError, match="No stored minute dataset"):
        resolve_update_dataset(
            store=store,
            receipt=_receipt(tmp_path),
            session=date(2025, 1, 13),
            version=DatasetVersion(uuid4()),
        )


def test_falls_back_to_unsuffixed_id(tmp_path: Path) -> None:
    store = Mock()
    version = DatasetVersion(uuid4())
    store.list_versions.side_effect = [[], [Mock(version=version)]]
    dataset_id, chosen = resolve_update_dataset(
        store=store,
        receipt=_receipt(tmp_path),
        session=date(2025, 1, 13),
    )
    assert dataset_id == DatasetId("market.ohlcv.minute.aapl")
    assert chosen == version
