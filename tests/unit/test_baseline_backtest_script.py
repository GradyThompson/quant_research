"""Validate the offline smoke runner's input hygiene."""

import json
from pathlib import Path

import pytest

from quant_research.backtesting.baseline import load_config


def test_valid_config(tmp_path: Path) -> None:
    config = {
        "store_root": "data",
        "datasets": [{"dataset_id": "market.ohlcv.daily", "version": "not-validated-here"}],
        "instruments": [{"security_id": "SEC-000001"}],
        "buy_security_id": "SEC-000001",
        "shares": 1,
        "initial_cash_usd": "10000",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_config(path)["shares"] == 1


@pytest.mark.parametrize("shares", [0, -1, 0.1, True])
def test_invalid_shares(tmp_path: Path, shares: object) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"datasets": [{}], "instruments": [{}], "shares": shares, "initial_cash_usd": "100"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shares"):
        load_config(path)


def test_invalid_cash(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"datasets": [{}], "instruments": [{}], "shares": 1, "initial_cash_usd": "NaN"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="initial_cash"):
        load_config(path)
