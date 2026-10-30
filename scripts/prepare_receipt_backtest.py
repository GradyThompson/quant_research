"""Prepare a pinned baseline backtest from a successful Massive ingestion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_research.backtesting.receipt import config_from_ingestion_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/massive_daily_pilot_receipt.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--buy-ticker", default="AAPL")
    parser.add_argument("--shares", type=int, default=1)
    parser.add_argument("--initial-cash-usd", default="10000")
    parser.add_argument("--output", type=Path, default=Path("data/dev/backtest_config.json"))
    args = parser.parse_args()
    config = config_from_ingestion_receipt(
        args.receipt,
        store_root=args.store_root,
        buy_ticker=args.buy_ticker,
        shares=args.shares,
        initial_cash_usd=args.initial_cash_usd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing config: {args.output}")
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pinned baseline config: {args.output}")
    print(
        f"Pinned datasets: {len(config['datasets'])}; selected security: {config['buy_security_id']}"
    )


if __name__ == "__main__":
    main()
