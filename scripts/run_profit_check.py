"""Precision-first barrier research, optionally with one-time holdout replay.

Default: 38 training sessions, embargo, 10 validation sessions, embargo,
10 reserved holdout sessions. Research results are not evidence of live fills.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from quant_research.model.profit_execution import ExecutionCosts
from quant_research.model.profit_research import ProfitPolicyRequirements, run_profit_research

from quant_research.data.storage import ParquetDatasetStore
from quant_research.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--training", type=int, default=38)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=10)
    parser.add_argument("--embargo", type=int, default=1)
    parser.add_argument("--training-stride", type=int, default=5)
    parser.add_argument("--evaluation-stride", type=int, default=62)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--minimum-trades", type=int, default=10)
    parser.add_argument("--minimum-precision-lower-bound", type=float, default=0.5)
    parser.add_argument("--minimum-opportunity-fraction", type=float, default=0.05)
    parser.add_argument("--initial-cash", type=float, default=10_000)
    parser.add_argument("--position-fraction", type=float, default=0.10)
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="One-time access to reserved holdout after the threshold is frozen",
    )
    args = parser.parse_args()
    output = args.output_dir or Path("data/dev/models") / f"{args.ticker.lower()}_profit_check"
    result = run_profit_research(
        ticker=args.ticker,
        receipt_path=resolve_project_path(args.receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        output_dir=resolve_project_path(output),
        evaluation_start=args.evaluation_start,
        training=args.training,
        validation=args.validation,
        holdout=args.holdout,
        embargo=args.embargo,
        training_stride=args.training_stride,
        evaluation_stride=args.evaluation_stride,
        ridge=args.ridge,
        upper_pct=args.upper_pct,
        lower_pct=args.lower_pct,
        horizon=args.horizon,
        costs=ExecutionCosts(args.slippage_bps, args.commission_per_share),
        requirements=ProfitPolicyRequirements(
            minimum_trades=args.minimum_trades,
            minimum_opportunity_fraction=args.minimum_opportunity_fraction,
            minimum_precision_lower_bound=args.minimum_precision_lower_bound,
        ),
        initial_cash=args.initial_cash,
        position_fraction=args.position_fraction,
        evaluate_holdout=args.evaluate_holdout,
    )
    info = result["validation"]
    score = info["scores"]
    print(
        f"{args.ticker}: training examples={result['training']['usable']}; "
        f"training sessions={result['training']['sessions']}"
    )
    print(
        f"Validation: decisions={score['decisions']}, positive={score['positive_net']}, "
        f"profit_AUC={score['profit_auc']}, log_loss={score['profit_log_loss']:.6f}"
    )
    policy = info["selected_policy"]
    if policy is None:
        print("NO TRADE POLICY: no threshold meets precision, opportunity and net-return gates")
        print("Reserved holdout NOT opened")
    else:
        print(
            f"Selected threshold={policy['threshold']:.2f}, "
            f"validation precision={policy['precision']:.1%}, "
            f"lower_bound={policy['precision_lower_bound']:.1%}, "
            f"n={policy['trades']}"
        )
        report = info["simulated_profit"]
        print(f"Validation replay: {report['trades']} trades, net P&L=${report['net_pnl']:.2f}")
    if result["holdout_opened"]:
        held = result["holdout"]["simulated_profit"]
        print(
            f"Holdout replay: trades={held['trades']}, "
            f"precision={held['precision']}, net P&L=${held['net_pnl']:.2f}, "
            f"return={held['return_pct']:.3f}%"
        )
    elif args.evaluate_holdout and policy is None:
        print("Holdout request skipped: no policy was approved on validation")
    print(f"Report: {output / 'profit_check.json'}")
    print("NOTICE: historical OHLC fill approximation; not live/executable P&L")


if __name__ == "__main__":
    main()
