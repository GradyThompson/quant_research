"""Build an offline baseline-backtest configuration from a pinned ingestion receipt.

No API calls and no writes. The caller chooses where to save the configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import UUID

from quant_research.core import DatasetId, SecurityId


def config_from_ingestion_receipt(
    receipt_path: Path,
    *,
    store_root: Path,
    buy_ticker: str = "AAPL",
    shares: int = 1,
    initial_cash_usd: str = "10000",
) -> dict[str, Any]:
    """Convert Change 43's receipt to the existing baseline-backtest config.

    The receipt is a *data-pinning record*, not a trusted source of prices or
    execution authorization. Dataset contents are validated on backtest read.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be a JSON object")
    rows = receipt.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("receipt must contain nonempty datasets")
    if not isinstance(buy_ticker, str) or not buy_ticker.strip():
        raise ValueError("buy_ticker must be nonempty")
    if type(shares) is not int or shares <= 0:
        raise ValueError("shares must be a positive integer")
    try:
        cash = Decimal(initial_cash_usd)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("initial_cash_usd must be a positive finite number") from exc
    if not cash.is_finite() or cash <= 0:
        raise ValueError("initial_cash_usd must be a positive finite number")

    datasets: list[dict[str, str]] = []
    instruments: list[dict[str, str]] = []
    seen_tickers: set[str] = set()
    seen_ids: set[str] = set()
    seen_datasets: set[str] = set()
    selected_sid: str | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each receipt dataset must be an object")
        try:
            ticker = row["ticker"]
            mic = row["mic"]
            sid = SecurityId(row["security_id"]).value
            dataset_id = DatasetId(row["dataset_id"]).value
            version = str(UUID(row["version"]))
            count = row["rows"]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid receipt dataset metadata") from exc
        if not isinstance(ticker, str) or not ticker or mic not in ("XNAS", "XNYS"):
            raise ValueError("receipt requires US ticker and supported MIC")
        if type(count) is not int or count < 1:
            raise ValueError("receipt dataset has no bars")
        if ticker in seen_tickers or sid in seen_ids or dataset_id in seen_datasets:
            raise ValueError("duplicate ticker, security, or dataset in receipt")
        seen_tickers.add(ticker)
        seen_ids.add(sid)
        seen_datasets.add(dataset_id)
        datasets.append({"dataset_id": dataset_id, "version": version})
        instruments.append({"security_id": sid})
        if ticker == buy_ticker.upper():
            selected_sid = sid
    if selected_sid is None:
        raise ValueError(f"buy ticker {buy_ticker} is absent from receipt")
    return {
        "store_root": str(store_root.expanduser().resolve()),
        "datasets": datasets,
        "instruments": instruments,
        "buy_security_id": selected_sid,
        "shares": shares,
        "initial_cash_usd": str(cash),
    }
