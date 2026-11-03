"""Run pinned security checks and shared-cash replay from approved signals."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quant_research.model.profit_execution import ExecutionCosts
from quant_research.model.profit_portfolio import PortfolioLimits, replay_shared_portfolio
from quant_research.model.profit_research import ProfitPolicyRequirements, run_profit_research

from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", required=True, help="Comma-separated list, e.g. AAPL,NVDA,MSFT"
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dev/models/profit_batch_01"))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--training-stride", type=int, default=5)
    parser.add_argument("--evaluation-stride", type=int, default=62)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--minimum-trades", type=int, default=10)
    parser.add_argument("--minimum-precision-lower-bound", type=float, default=0.5)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--initial-cash", type=float, default=10_000)
    parser.add_argument("--position-fraction", type=float, default=0.10)
    parser.add_argument("--max-concurrent", type=int, default=3)
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]
    if not tickers or any(not t.isalpha() or not t.isascii() or not t.isupper() for t in tickers):
        raise ValueError("Tickers must be comma-separated uppercase ASCII symbols")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate tickers")
    root = resolve_project_path(args.output_dir)
    if root.exists():
        raise FileExistsError(f"Batch directory already exists: {root}")
    # Fail-fast on errors: do not silently report a partial batch as complete.
    store = ParquetDatasetStore(resolve_project_path(args.store_root))
    summary = []
    validation_ledgers: dict[str, list[dict[str, object]]] = {}
    holdout_ledgers: dict[str, list[dict[str, object]]] = {}
    for ticker in tickers:
        result = run_profit_research(
            ticker=ticker,
            receipt_path=resolve_project_path(args.receipt),
            store=store,
            output_dir=root / ticker.lower(),
            evaluation_start=args.evaluation_start,
            training_stride=args.training_stride,
            evaluation_stride=args.evaluation_stride,
            ridge=args.ridge,
            costs=ExecutionCosts(args.slippage_bps, args.commission_per_share),
            requirements=ProfitPolicyRequirements(
                minimum_trades=args.minimum_trades,
                minimum_precision_lower_bound=args.minimum_precision_lower_bound,
            ),
            evaluate_holdout=args.evaluate_holdout,
            initial_cash=args.initial_cash,
            position_fraction=args.position_fraction,
        )
        policy = result["validation"]["selected_policy"]
        replay = result["holdout"] if result["holdout_opened"] else None
        if policy is not None:
            validation_ledgers[ticker] = result["validation"]["simulated_profit"]["ledger"]
            if replay is not None:
                holdout_ledgers[ticker] = replay["simulated_profit"]["ledger"]
        summary.append(
            {
                "ticker": ticker,
                "validation_decisions": result["validation"]["scores"]["decisions"],
                "validation_profit_auc": result["validation"]["scores"]["profit_auc"],
                "selected_threshold": policy["threshold"] if policy else None,
                "validation_positive_precision": policy["precision"] if policy else None,
                "holdout_opened": result["holdout_opened"],
                "holdout_net_pnl": (replay["simulated_profit"]["net_pnl"] if replay else None),
                "holdout_trades": (replay["simulated_profit"]["trades"] if replay else None),
            }
        )
        status = "approved" if policy else "abstained"
        print(f"{ticker}: {status}; holdout={result['holdout_opened']}")
    root.mkdir(parents=True, exist_ok=True)
    limits = PortfolioLimits(
        initial_cash=args.initial_cash,
        position_fraction=args.position_fraction,
        max_concurrent_positions=args.max_concurrent,
        commission_per_share=args.commission_per_share,
    )
    validation_portfolio = replay_shared_portfolio(validation_ledgers, limits=limits)
    holdout_portfolio = (
        replay_shared_portfolio(holdout_ledgers, limits=limits)
        if args.evaluate_holdout and holdout_ledgers
        else None
    )
    (root / "portfolio_validation.json").write_text(
        json.dumps(validation_portfolio, indent=2, sort_keys=True) + "\n"
    )
    if holdout_portfolio is not None:
        (root / "portfolio_holdout.json").write_text(
            json.dumps(holdout_portfolio, indent=2, sort_keys=True) + "\n"
        )
    (root / "batch_summary.json").write_text(
        json.dumps(
            {
                "tickers": summary,
                "portfolio_validation": "portfolio_validation.json",
                "portfolio_holdout": (
                    "portfolio_holdout.json" if holdout_portfolio is not None else None
                ),
                "warning": (
                    "Portfolio validation was selected on validation data: not independent. "
                    "Historical OHLC fills are not live executable orders."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {args.output_dir / 'batch_summary.json'}")
    print(
        f"Shared-cash validation: {validation_portfolio['trades']} trades, "
        f"net P&L=${validation_portfolio['net_pnl']:.2f} (selected on validation)"
    )
    if holdout_portfolio is not None:
        print(
            f"Shared-cash holdout: {holdout_portfolio['trades']} trades, "
            f"net P&L=${holdout_portfolio['net_pnl']:.2f}, "
            f"precision={holdout_portfolio['precision']}"
        )
    print("NOTICE: OHLC historical fill approximation, not live executable P&L")


if __name__ == "__main__":
    main()
