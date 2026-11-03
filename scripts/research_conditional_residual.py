"""P5-A conditional ETF-proxy residual reversal using verified minute summaries.

EXPLORATORY ONLY: no split-adjusted stock prices or historical security-type
classification. Results are not broker-executable, fully hedged portfolio P&L.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.ingestion.minute_market_summary import (
    SummaryJob,
    summarize_market_job,
)
from quant_research.model.conditional_residual import (
    DailyPrice,
    generate_residual_events,
    score_residual_events,
)
from quant_research.model.opening_range_history import protected_research_days
from quant_research.paths import resolve_project_path

# These are tradeable proxies for sector exposure, NOT a verified historical
# GICS/SIC classification of each issuer. Selection uses only prior returns.
ETF_FACTORS = (
    "SPY",
    "QQQ",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)


def _verified_source(*, manifest_root: Path, day: date) -> tuple[Path, str]:
    """Require one pinned source revision with identical SHA256 and length."""
    manifest = json.loads((manifest_root / f"{day}.json").read_text(encoding="utf-8"))
    key = f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz"
    revisions = manifest.get("revisions")
    if (
        manifest.get("format") != 1
        or manifest.get("bucket") != "flatfiles"
        or manifest.get("key") != key
        or not isinstance(revisions, list)
        or len(revisions) != 1
    ):
        raise ValueError(f"invalid pinned Massive manifest for {day}")
    revision = revisions[0]
    path = Path(revision["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing compressed source for {day}: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(data)
    if sha.hexdigest() != revision["sha256"] or path.stat().st_size != revision["bytes"]:
        raise ValueError(f"source verification failed: {day}")
    return path, sha.hexdigest()


def run_research(
    *,
    start: date,
    end: date,
    validation_start: date,
    manifest_root: Path,
    original_receipt_path: Path,
    output_dir: Path,
    summary_cache_root: Path = Path("data/dev/massive_flat_files/market_summaries"),
    factors: tuple[str, ...] = ETF_FACTORS,
    holdings: tuple[int, ...] = (3, 5, 10),
    costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    formation_sessions: int = 5,
    estimation_sessions: int = 60,
    max_universe: int = 300,
    max_per_side: int = 3,
    progress_every: int = 10,
    bootstrap_repeats: int = 1000,
) -> dict[str, Any]:
    """Read only permitted sessions and never overwrite a previous report."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite research report: {output_dir}")
    if (
        start >= validation_start
        or validation_start > end
        or progress_every < 1
        or bootstrap_repeats < 2
        or len(set(factors)) != len(factors)
        or "SPY" not in factors
        or not holdings
        or any(h < 1 for h in holdings)
        or not costs_bps
        or any(cost < 0 for cost in costs_bps)
    ):
        raise ValueError("invalid P5 experiment arguments")
    receipt_bytes = original_receipt_path.read_bytes()
    protected = protected_research_days(json.loads(receipt_bytes))
    cal = xcals.get_calendar("XNYS")
    all_days = [
        stamp.date() for stamp in cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    ]
    eligible = [day for day in all_days if day >= validation_start and day not in protected]
    if not eligible:
        raise ValueError("no permissible validation days")
    panel: dict[date, dict[str, DailyPrice]] = {}
    sources: list[dict[str, str]] = []
    cache_hits = 0
    began = monotonic()
    # Iterate over exchange sessions, preserving protected/missing sessions as
    # explicit gaps so a lookback window cannot silently bridge a holdout.
    for i, day in enumerate(all_days, 1):
        if day in protected:
            panel[day] = {}
            continue
        path, sha = _verified_source(manifest_root=manifest_root, day=day)
        sources.append({"day": day.isoformat(), "sha256": sha})
        stamp = pd.Timestamp(day)
        job: SummaryJob = (
            day,
            path,
            sha,
            summary_cache_root,
            cal.session_open(stamp).to_pydatetime(),
            cal.session_close(stamp).to_pydatetime(),
        )
        _, summary, hit = summarize_market_job(job)
        cache_hits += int(hit)
        panel[day] = {
            ticker: DailyPrice(
                opening=row.opening_open,
                closing=row.last_close,
                dollar_volume=row.dollar_volume,
                close_eligible=row.close_eligible,
            )
            for ticker, row in summary.items()
        }
        if i % progress_every == 0 or i == len(all_days):
            print(
                f"P5 cached summaries {i}/{len(all_days)}, hits={cache_hits}, "
                f"elapsed={monotonic() - began:.1f}s",
                flush=True,
            )
    # The return engine only uses data through formation day to determine entry
    # decisions; future opens are accessed solely to label selected events.
    events, diagnostics = generate_residual_events(
        days=all_days,
        panel=panel,
        factor_tickers=factors,
        validation_start=validation_start,
        formation_sessions=formation_sessions,
        estimation_sessions=estimation_sessions,
        holding_sessions=holdings,
        max_universe=max_universe,
        max_per_side=max_per_side,
        progress=lambda day, index, size: print(
            f"P5 signal evaluation {index}/{size} ({day})", flush=True
        ),
    )
    scores: dict[str, object] = {}
    for horizon in holdings:
        for side in ("negative_long", "positive_short"):
            for regime in ("all", "range", "trend"):
                scores[f"{horizon}d_{side}_{regime}"] = {
                    str(cost): score_residual_events(
                        events=events,
                        eligible_days=eligible,
                        horizon=horizon,
                        side=side,
                        regime=regime,
                        roundtrip_cost_bps=cost,
                        bootstrap_repeats=bootstrap_repeats,
                        slots=max_per_side,
                    )
                    for cost in costs_bps
                }
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_family": "P5-A-conditional-ETF-proxy-residual-reversal",
        "research_only": True,
        "not_trading_ready": True,
        "holdout_accessed": False,
        "original_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "protected_dates_excluded_before_read": sorted(
            day.isoformat() for day in protected if start <= day <= end
        ),
        "source_revisions": sources,
        "configuration": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "eligible_days": len(eligible),
            "factors": list(factors),
            "formation_sessions": formation_sessions,
            "estimation_sessions": estimation_sessions,
            "holding_sessions": list(holdings),
            "costs_bps": list(costs_bps),
            "max_universe": max_universe,
            "max_per_side": max_per_side,
            "shock_z_threshold": 2.0,
            "range_efficiency_threshold": 0.25,
            "entry": "next regular session opening-bar open (proxy)",
            "exit": "opening-bar open after N held exchange sessions (proxy)",
        },
        "market_summary_cache_hits": cache_hits,
        "candidate_diagnostics": diagnostics,
        "total_scored_events": len(events),
        "results": scores,
        "critical_limitations": [
            "Minute summary ticker is NOT a verified point-in-time common-stock security identity.",
            "Prices are NOT split or dividend adjusted; discontinuities above 30% are rejected.",
            "Missing future prices, potential delistings, or halts censor returns; see diagnostics.",
            "ETF factor chosen from lagged 60-session correlation is a proxy, NOT industry classification.",
            "ETFs and stock leg are charged separately but spread/borrow are not verified.",
            "Overlapping multi-day positions are NOT capital-constrained portfolio P&L.",
            "Opening minute prices are bar proxies, not broker-executable quotes.",
            "Previously studied 2023-2024 data are DEVELOPMENT observations, NOT fresh holdout.",
            "Several related horizons/sides/regimes are examined; intervals are exploratory.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, by_cost in scores.items():
        metric = by_cost[str(costs_bps[0])]
        print(
            f"{name}: trades={metric['trades']} "
            f"cost={costs_bps[0]:g}bps "
            f"net/entry_session={metric['net_bps_per_entry_session']:+.3f}bps "
            "[DIAGNOSTIC ONLY]",
            flush=True,
        )
    print(f"Report: {output_dir / 'validation.json'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--validation-start", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/minute_manifests")
    )
    parser.add_argument(
        "--original-receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument(
        "--summary-cache-root",
        type=Path,
        default=Path("data/dev/massive_flat_files/market_summaries"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-universe", type=int, default=300)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    run_research(
        start=args.start,
        end=args.end,
        validation_start=args.validation_start,
        manifest_root=resolve_project_path(args.manifest_root),
        original_receipt_path=resolve_project_path(args.original_receipt),
        output_dir=resolve_project_path(args.output_dir),
        summary_cache_root=resolve_project_path(args.summary_cache_root),
        progress_every=args.progress_every,
        max_universe=args.max_universe,
        bootstrap_repeats=args.bootstrap_repeats,
    )


if __name__ == "__main__":
    main()
