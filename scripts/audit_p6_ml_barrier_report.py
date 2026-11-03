"""Read-only P6 OOF audit: paired model differences, selection, and tails.

Uses the saved validation.json only. Does not access market-price data,
train models, or open the protected holdout. Block-bootstrap intervals are
exploratory and do not correct for repeated research/model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _scenario(model: dict[str, Any], cost: float) -> dict[str, Any]:
    scenarios: dict[str, Any] = model["cost_scenarios"]
    matches = [value for key, value in scenarios.items() if float(key) == cost]
    if len(matches) != 1:
        raise ValueError(f"expected one scenario for cost {cost:g} bps")
    return matches[0]


def _block_ci(
    values: list[float],
    *,
    seed: int = 2026,
    block_length: int = 5,
    bootstrap_samples: int = 2000,
) -> list[float]:
    if not values:
        raise ValueError("no daily returns")
    if block_length < 1 or bootstrap_samples < 2:
        raise ValueError("invalid bootstrap settings")
    if len(values) == 1:
        return [values[0], values[0]]
    count = len(values)
    rng = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(bootstrap_samples):
        sample: list[float] = []
        while len(sample) < count:
            start = rng.randrange(count)
            sample.extend(values[(start + offset) % count] for offset in range(block_length))
        sample_means.append(sum(sample[:count]) / count)
    sample_means.sort()
    return [
        sample_means[int(0.025 * (bootstrap_samples - 1))],
        sample_means[int(0.975 * (bootstrap_samples - 1))],
    ]


def audit(report: dict[str, Any], *, cost_bps: float = 4.0) -> dict[str, Any]:
    """Diagnose the *unchanged* OOF trade selections for each model."""
    if (
        report.get("schema_version") != 1
        or report.get("model") != "P6-ML-3outcome-cross-sectional-barrier"
    ):
        raise ValueError("not a P6 validation report")
    models: dict[str, dict[str, Any]] = report["pooled"]
    if "math" not in models or "direct" not in models or "ensemble" not in models:
        raise ValueError("missing required P6 models")
    slots = int(report["configuration"]["positions"])
    if slots < 1:
        raise ValueError("invalid portfolio slot count")
    per_model: dict[str, dict[str, Any]] = {}
    daily_by_model: dict[str, dict[str, float]] = {}
    selected_by_model: dict[str, set[tuple[str, str]]] = {}
    days: list[str] | None = None
    for name, model in models.items():
        source = _scenario(model, cost_bps)
        daily = {str(day): float(value) for day, value in source["daily_bps"].items()}
        if not daily or not all(math.isfinite(x) for x in daily.values()):
            raise ValueError("missing/nonfinite OOF session returns")
        if days is None:
            days = sorted(daily)
        elif sorted(daily) != days:
            raise ValueError("model sessions differ; paired comparison invalid")
        trades = source["trades"]
        if len(trades) != int(source["trades_count"]):
            raise ValueError("trade count mismatch")
        identifiers = {(str(t["day"]), str(t["ticker"])) for t in trades}
        if len(identifiers) != len(trades):
            raise ValueError("duplicate selected trades")
        if any(day not in daily for day, _ in identifiers):
            raise ValueError("trade outside validation sessions")
        reconstruction = dict.fromkeys(daily, 0.0)
        nets: list[float] = []
        for trade in trades:
            net = float(trade["net_bps"])
            if not math.isfinite(net):
                raise ValueError("nonfinite trade returns")
            reconstruction[str(trade["day"])] += net / slots
            nets.append(net)
        if any(abs(reconstruction[day] - daily[day]) > 1e-5 for day in daily):
            raise ValueError("saved net returns do not reconstruct from trades")
        top_5 = sorted(nets, reverse=True)[:5]
        adjusted_mean = (sum(daily.values()) - sum(top_5) / slots) / len(daily)
        per_model[name] = {
            "scored": int(model["scored"]),
            "brier": float(model["brier"]),
            "log_loss": float(model["log_loss"]),
            "sessions": len(daily),
            "trades": len(trades),
            "slot_utilization": len(trades) / (len(daily) * slots),
            "net_bps_per_session": statistics.mean(daily.values()),
            "mean_net_bps_per_trade": statistics.mean(nets) if nets else None,
            "median_net_bps_per_trade": statistics.median(nets) if nets else None,
            "without_best_5_bps_per_original_session": adjusted_mean,
            "five_day_circular_block_ci": _block_ci([daily[d] for d in sorted(daily)]),
        }
        daily_by_model[name] = daily
        selected_by_model[name] = identifiers
    assert days is not None
    pairs: dict[str, Any] = {}
    for first, second in (("direct", "math"), ("direct", "ensemble"), ("ensemble", "math")):
        delta = [daily_by_model[first][day] - daily_by_model[second][day] for day in days]
        a, b = selected_by_model[first], selected_by_model[second]
        pairs[f"{first}_minus_{second}"] = {
            "mean_incremental_bps_per_session": statistics.mean(delta),
            "five_day_circular_block_ci": _block_ci(delta, seed=2035),
            "shared_selected_trades": len(a & b),
            "selected_trade_jaccard": len(a & b) / len(a | b) if a or b else None,
        }
    return {
        "research_only": True,
        "read_only": True,
        "source": "P6 OOF validation report; no historical rescan",
        "cost_bps_additional_to_commissions": cost_bps,
        "models": per_model,
        "paired": pairs,
        "caveats": [
            "Five-day circular block bootstrap is exploratory; models were compared repeatedly.",
            "The removal of top trades is a concentration diagnostic, not a reoptimized strategy.",
            "Price-proxy outcomes; no historical quote, broker execution or selection calibration.",
            "Net/session assumes reserved equal portfolio slots and holds cash in unused slots.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.report / "validation.json" if args.report.is_dir() else args.report
    data = audit(json.loads(source.read_text(encoding="utf-8")), cost_bps=args.cost_bps)
    for name, row in data["models"].items():
        print(
            f"{name}: trades={row['trades']}/{row['sessions']} sessions, "
            f"utilization={row['slot_utilization']:.1%}, "
            f"net/session={row['net_bps_per_session']:+.3f}bps, "
            f"block CI={row['five_day_circular_block_ci']}"
        )
    for name, row in data["paired"].items():
        print(
            f"{name}: delta/session={row['mean_incremental_bps_per_session']:+.3f}bps, "
            f"block CI={row['five_day_circular_block_ci']}; "
            f"shared trades={row['shared_selected_trades']}"
        )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Audit: {args.output}")


if __name__ == "__main__":
    main()
