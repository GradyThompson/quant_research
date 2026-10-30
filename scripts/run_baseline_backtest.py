"""CLI entry point for an offline pinned US-equity baseline backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_research.backtesting.baseline import load_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/reports/baseline_backtest.json"))
    args = parser.parse_args()
    report = run(load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved offline backtest report to {args.output}")
    print(
        f"Fills: {len(report['fills'])}; total return: {report['total_return']}; max drawdown: {report['max_drawdown']}"
    )


if __name__ == "__main__":
    main()
