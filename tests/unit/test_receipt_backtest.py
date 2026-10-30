"""Check ingestion-receipt conversion without downloading market data."""

import json
from pathlib import Path

import pytest

from quant_research.backtesting.receipt import config_from_ingestion_receipt

VERSION = "6fc6c71a-4f05-4a56-b683-9e1681af4bf5"


def make_receipt(path: Path, datasets: list[dict] | None = None) -> None:
    if datasets is None:
        datasets = [
            {
                "ticker": "AAPL",
                "mic": "XNAS",
                "security_id": "SEC-14290302951544531124",
                "dataset_id": "market.ohlcv.daily.aapl",
                "version": VERSION,
                "rows": 20,
            },
            {
                "ticker": "MSFT",
                "mic": "XNAS",
                "security_id": "SEC-01267475381686077525",
                "dataset_id": "market.ohlcv.daily.msft",
                "version": VERSION,
                "rows": 20,
            },
        ]
    path.write_text(json.dumps({"security_master_snapshot": VERSION, "datasets": datasets}))


def test_builds_pinned_config(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    make_receipt(receipt)
    config = config_from_ingestion_receipt(
        receipt, store_root=tmp_path, buy_ticker="MSFT", shares=2
    )
    assert config["buy_security_id"] == "SEC-01267475381686077525"
    assert config["store_root"] == str(tmp_path.resolve())
    assert len(config["datasets"]) == 2
    assert config["datasets"][0]["version"] == VERSION
    assert config["shares"] == 2


def test_missing_selected_ticker(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    make_receipt(receipt)
    with pytest.raises(ValueError, match="absent"):
        config_from_ingestion_receipt(receipt, store_root=tmp_path, buy_ticker="NVDA")


def test_duplicate_security_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    make_receipt(receipt)
    data = json.loads(receipt.read_text())
    data["datasets"][1]["security_id"] = data["datasets"][0]["security_id"]
    make_receipt(receipt, data["datasets"])
    with pytest.raises(ValueError, match="duplicate"):
        config_from_ingestion_receipt(receipt, store_root=tmp_path)


def test_rejects_zero_rows(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    make_receipt(receipt)
    data = json.loads(receipt.read_text())
    data["datasets"][0]["rows"] = 0
    make_receipt(receipt, data["datasets"])
    with pytest.raises(ValueError, match="no bars"):
        config_from_ingestion_receipt(receipt, store_root=tmp_path)


@pytest.mark.parametrize("shares,cash", [(0, "10000"), (1, "0"), (1, "NaN")])
def test_invalid_capital_inputs(tmp_path: Path, shares: int, cash: str) -> None:
    receipt = tmp_path / "receipt.json"
    make_receipt(receipt)
    with pytest.raises(ValueError):
        config_from_ingestion_receipt(
            receipt, store_root=tmp_path, shares=shares, initial_cash_usd=cash
        )
