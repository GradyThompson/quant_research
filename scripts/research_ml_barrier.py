"""P6: purged chronological ML barrier research and top-N price-proxy evaluation.

Development-only. Train mathematical, direct, touch/direction, timeout and
fixed-weight ensemble heads; portfolio slots earn cash if not selected.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from quant_research.model.ml_barrier import BarrierRow, row_from_json
from quant_research.model.ml_barrier_ensemble import forecast_all, score_predictions


def load_dataset(root: Path) -> tuple[dict[str, Any], list[BarrierRow]]:
    manifest_path = root / "manifest.json"
    rows_path = root / "rows.jsonl.gz"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protected_holdout_opened") is not False or manifest.get("schema_version") != 1:
        raise ValueError("dataset lacks protected-holdout certification")
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != manifest["rows_sha256"]:
        raise ValueError("barrier dataset hash does not match manifest")
    protected = {date.fromisoformat(v) for v in manifest["protected_dates"]}
    rows: list[BarrierRow] = []
    with gzip.open(rows_path, "rt", encoding="utf-8") as source:
        for line in source:
            row = row_from_json(json.loads(line))
            if row.day in protected or row.resolved_at.date() in protected:
                raise ValueError("protected date in dataset")
            if row.resolved_at.date() != row.day or row.entry_at.date() != row.day:
                # Timestamp is UTC; RTH exit can be next UTC day? No, US stocks
                # close before midnight UTC; this still fails closed.
                raise ValueError("non-intraday or chronologically invalid label")
            rows.append(row)
    if len(rows) != manifest["counts"]["rows"]:
        raise ValueError("dataset row count mismatch")
    identifiers = {(row.day, row.ticker) for row in rows}
    if len(identifiers) != len(rows):
        raise ValueError("duplicate ticker/decision-day observation")
    return manifest, rows


def chronological_folds(
    rows: list[BarrierRow],
    *,
    validation_start: date,
    min_train_sessions: int = 60,
    validation_sessions: int = 20,
    embargo_sessions: int = 1,
) -> list[tuple[list[int], list[int]]]:
    if min_train_sessions < 10 or validation_sessions < 1 or embargo_sessions < 1:
        raise ValueError("invalid chronological folds")
    sessions = sorted({row.day for row in rows})
    if len(sessions) < min_train_sessions + embargo_sessions + validation_sessions:
        raise ValueError("insufficient historical sessions for a meaningful walk-forward")
    groups: dict[date, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.day].append(index)
    first_val = next(
        (i for i, day in enumerate(sessions) if day >= validation_start), len(sessions)
    )
    start = max(min_train_sessions + embargo_sessions, first_val)
    result: list[tuple[list[int], list[int]]] = []
    for index in range(start, len(sessions), validation_sessions):
        training = [idx for day in sessions[: index - embargo_sessions] for idx in groups[day]]
        validation = [
            idx for day in sessions[index : index + validation_sessions] for idx in groups[day]
        ]
        if training and validation:
            assert max(rows[i].resolved_at for i in training) < min(
                rows[i].entry_at for i in validation
            )
            result.append((training, validation))
    if not result:
        raise ValueError("no scored validation folds")
    return result


def _session_ci(daily: dict[str, float], *, seed: int, replicates: int) -> list[float]:
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    sample = np.asarray(list(daily.values()), dtype=float)
    if len(sample) < 2:
        return [float(sample.mean()), float(sample.mean())]
    rng = np.random.default_rng(seed)
    boot = rng.choice(sample, size=(replicates, len(sample)), replace=True).mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def run_research(
    *,
    dataset: Path,
    output_dir: Path,
    validation_sessions: int = 20,
    min_train_sessions: int = 60,
    positions: int = 3,
    edge_margin_bps: float = 2.0,
    costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    seed: int = 2026,
    notional_dollars: float = 10_000.0,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite research results: {output_dir}")
    if not costs_bps or costs_bps[0] < 0 or edge_margin_bps < 0 or positions < 1:
        raise ValueError("invalid trading assumptions")
    manifest, rows = load_dataset(dataset)
    validation_start = date.fromisoformat(manifest["validation_start"])
    folds = chronological_folds(
        rows,
        validation_start=validation_start,
        validation_sessions=validation_sessions,
        min_train_sessions=min_train_sessions,
    )
    fold_results: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []
    for fold_no, (train_idx, val_idx) in enumerate(folds, 1):
        training = [rows[i] for i in train_idx]
        validation = [rows[i] for i in val_idx]
        computed = forecast_all(training, validation, seed=seed + fold_no)
        scores = score_predictions(
            rows=validation,
            pred=computed,
            cost_bps=costs_bps,
            positions=positions,
            edge_margin_bps=edge_margin_bps,
            notional_dollars=notional_dollars,
        )
        for model, by_model in scores.items():
            record = pooled.setdefault(
                model,
                {
                    "weighted_brier": 0.0,
                    "weighted_log_loss": 0.0,
                    "n": 0,
                    "cost_scenarios": {str(c): {"daily_bps": {}, "trades": []} for c in costs_bps},
                },
            )
            record["weighted_brier"] += by_model["brier"] * len(validation)
            record["weighted_log_loss"] += by_model["log_loss"] * len(validation)
            record["n"] += len(validation)
            for cost in costs_bps:
                source = by_model["cost_scenarios"][str(cost)]
                target = record["cost_scenarios"][str(cost)]
                overlap = set(target["daily_bps"]) & set(source["daily_bps"])
                if overlap:
                    raise ValueError("duplicate OOF validation sessions")
                target["daily_bps"].update(source["daily_bps"])
                target["trades"].extend(source["trades_detail"])
        for idx, row in enumerate(validation):
            predictions.append(
                {
                    "day": row.day.isoformat(),
                    "ticker": row.ticker,
                    "outcome": row.outcome,
                    "gross_return_bps": row.gross_return_bps,
                    "probabilities": {
                        key: [float(v) for v in p[idx]] for key, p in computed.probabilities.items()
                    },
                    "timeout_prediction_bps": float(computed.timeout_bps[idx]),
                }
            )
        fold_results.append(
            {
                "fold": fold_no,
                "train_sessions": len({r.day for r in training}),
                "validation_sessions": len({r.day for r in validation}),
                "train_last": max(r.day for r in training).isoformat(),
                "validation_first": min(r.day for r in validation).isoformat(),
                "validation_last": max(r.day for r in validation).isoformat(),
                "train_rows": len(training),
                "validation_rows": len(validation),
                "scores": scores,
            }
        )
        print(
            f"P6 fold {fold_no}/{len(folds)} training={len(training)} validation={len(validation)}",
            flush=True,
        )
    for metric in pooled.values():
        n = metric.pop("n")
        metric["brier"] = metric.pop("weighted_brier") / n
        metric["log_loss"] = metric.pop("weighted_log_loss") / n
        metric["scored"] = n
        for values in metric["cost_scenarios"].values():
            daily = values["daily_bps"]
            values["trades_count"] = len(values["trades"])
            values["net_bps_per_session"] = sum(daily.values()) / len(daily)
            values["session_bootstrap_ci"] = _session_ci(
                daily,
                seed=seed,
                replicates=bootstrap_replicates,
            )
    # Paired daily-difference confidence intervals preserve session dependence.
    paired_vs_math: dict[str, Any] = {}
    baseline_daily = pooled["math"]["cost_scenarios"][str(costs_bps[0])]["daily_bps"]
    for model in pooled:
        if model == "math":
            continue
        candidate_daily = pooled[model]["cost_scenarios"][str(costs_bps[0])]["daily_bps"]
        if set(baseline_daily) != set(candidate_daily):
            raise ValueError("model validation sessions do not align")
        paired = {day: candidate_daily[day] - baseline_daily[day] for day in baseline_daily}
        paired_vs_math[model] = {
            "mean_incremental_bps_per_session": sum(paired.values()) / len(paired),
            "session_bootstrap_ci": _session_ci(
                paired, seed=seed + 17, replicates=bootstrap_replicates
            ),
        }
    report = {
        "schema_version": 1,
        "research_only": True,
        "holdout_accessed": False,
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "model": "P6-ML-3outcome-cross-sectional-barrier",
        "configuration": {
            "upper_pct": 0.005,
            "lower_pct": 0.02,
            "horizon_minutes": 60,
            "decision_clock": "10:01 Eastern after 10:00 bar becomes available",
            "entry": "first complete-minute open strictly after decision, typically 10:02 Eastern",
            "costs_bps": costs_bps,
            "selection_cost_bps": costs_bps[0],
            "positions": positions,
            "edge_margin_bps": edge_margin_bps,
            "validation_sessions_per_fold": validation_sessions,
            "embargo_sessions": 1,
            "ensemble_weights": {
                "math": 0.15,
                "decomposed": 0.25,
                "correction": 0.25,
                "direct": 0.35,
            },
            "notional_dollars_per_position": notional_dollars,
            "execution_costs_bps_are_additional_to_ibkr_commissions": True,
            "timeout_model": "Ridge conditional on neither, clipped +/-200bps",
            "lower_loss": "historical conditional arithmetic stop loss, floor -200bps",
        },
        "folds": fold_results,
        "pooled": pooled,
        "paired_vs_math": paired_vs_math,
        "limitations": [
            "M2 proxy is morning realized-volatility mathematical price-zero first passage, NOT full M2.5 seasonal model.",
            "LightGBM and logistic probabilities are NOT post-hoc calibrated; inspect reliability.",
            "Pro Tiered share-price-aware estimated commissions, plus flat additional spread/slippage scenarios; not verified IBKR fills.",
            "Same-session three-slot price proxies, no open order simulation or market impact.",
            "Historical Massive CS reference is retrospective; effective reference publication not established.",
            "Corporate-action ex-dates excluded and only same-session returns labeled.",
            "Development-era 2023/2024 already influenced previous strategy research.",
            "Ensemble weights and thresholds fixed a priori; five alternatives are multiple research comparisons.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with gzip.open(output_dir / "oof_predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    for model, outcome in pooled.items():
        score = outcome["cost_scenarios"][str(costs_bps[0])]
        print(
            f"{model}: scored={outcome['scored']} trades={score['trades_count']} "
            f"net/session={score['net_bps_per_session']:+.3f}bps "
            f"CI={score['session_bootstrap_ci']} [PROXY ONLY]",
            flush=True,
        )
    print(f"Report: {output_dir / 'validation.json'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-sessions", type=int, default=60)
    parser.add_argument("--validation-sessions", type=int, default=20)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--edge-margin-bps", type=float, default=2.0)
    parser.add_argument("--additional-cost-bps", type=float, action="append", dest="costs")
    parser.add_argument("--notional-dollars", type=float, default=10000.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    run_research(
        dataset=args.dataset,
        output_dir=args.output_dir,
        validation_sessions=args.validation_sessions,
        min_train_sessions=args.min_train_sessions,
        positions=args.positions,
        edge_margin_bps=args.edge_margin_bps,
        costs_bps=tuple(args.costs) if args.costs else (4.0, 8.0, 12.0),
        bootstrap_replicates=args.bootstrap_replicates,
        notional_dollars=args.notional_dollars,
    )


if __name__ == "__main__":
    main()
