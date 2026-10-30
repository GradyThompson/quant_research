"""Independently verify a pinned, public buy-once baseline backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_research.backtesting.baseline import load_config, run
from quant_research.backtesting.verification import verify_baseline_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--replay", action="store_true", help="Rerun and compare complete deterministic report"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify_baseline_report(config, report)
    if args.replay:
        replay = run(config)
        if replay != report:
            raise ValueError("replayed baseline report differs from saved report")
    print("PASS: execution timing, fill accounting, equity, returns and drawdown")
    if args.replay:
        print("PASS: deterministic pinned-data replay")


if __name__ == "__main__":
    main()
