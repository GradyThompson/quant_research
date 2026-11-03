"""Research ORB across many dates using *verified local* Massive minute flat files.

No data purchase or broker connection. Explicit development dates only. A
pre-existing original receipt is REQUIRED: protected historical sessions and
its original evaluation date are excluded BEFORE any market file is opened.

Usage: provide cache/manifest dirs already populated by Massive downloader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.ingestion.minute_research_extract import (
    load_or_parse_selected_minutes,
)
from quant_research.data.ingestion.minute_research_extract import (
    parse_selected_minutes_fast as parse_selected_minutes,
)
from quant_research.model.first_passage_validation import (
    MinuteBar,
    Session,
)
from quant_research.model.opening_range_history import (
    evaluate_opening_range_history,
    protected_research_days,
)
from quant_research.paths import resolve_project_path


def _verified_source(*, manifest_root: Path, day: date) -> tuple[Path, dict[str, object]]:
    """Match a cached gzip against its pinned Massive SHA-256 manifest."""
    manifest_file = manifest_root / f"{day}.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"missing dated Massive manifest: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_key = f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz"
    if (
        manifest.get("format") != 1
        or manifest.get("bucket") != "flatfiles"
        or manifest.get("key") != expected_key
    ):
        raise ValueError(f"Massive manifest source mismatch: {day}")
    revisions = manifest.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError(f"expected exactly one pinned revision for {day}")
    revision: dict[str, object] = revisions[0]
    path = Path(str(revision["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"pinned gzip missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    if digest.hexdigest() != revision["sha256"] or path.stat().st_size != revision["bytes"]:
        raise ValueError(f"pinned Massive gzip hash/size mismatch: {day}")
    return path, revision


def _to_session(day: date, records: list[dict[str, object]]) -> Session:
    bars = tuple(
        MinuteBar(
            timestamp=row["timestamp"],
            available_at=row["available_at"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in records
    )
    return Session(day, bars)


def run_flat_history_research(
    *,
    tickers: tuple[str, ...],
    factor_ticker: str,
    manifest_root: Path,
    original_receipt_path: Path,
    start: date,
    end: date,
    validation_start: date,
    output_dir: Path,
    horizons: tuple[int, ...] = (30, 60),
    costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    publication_lag_seconds: int = 60,
    bootstrap_replicates: int = 2000,
    extract_cache_root: Path | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Stream verified market files only from an explicit permitted date span."""
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing report overwrite: {output_dir}")
    if not start < validation_start <= end:
        raise ValueError("require start < validation_start <= end")
    if publication_lag_seconds < 1:
        raise ValueError("publication lag must be positive")
    if not tickers or len(set(tickers)) != len(tickers) or factor_ticker in tickers:
        raise ValueError("tickers must be nonempty, unique and distinct from factor")
    if any(
        not ticker.isascii() or not ticker.isalpha() or not ticker.isupper()
        for ticker in (*tickers, factor_ticker)
    ):
        raise ValueError("tickers must be uppercase ASCII letters")
    # Explicit guard: receipt is mandatory and must support original anchor.
    original = json.loads(original_receipt_path.read_text(encoding="utf-8"))
    protected = protected_research_days(original)
    calendar = xcals.get_calendar("XNYS")
    days = tuple(
        stamp.date() for stamp in calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    )
    authorized = tuple(day for day in days if day not in protected)
    if not authorized:
        raise ValueError("no authorized dates in requested interval")
    # Check all source manifests before loading any large market files; fail
    # closed on missing sessions rather than silently selecting lucky dates.
    pinned: list[tuple[date, Path, dict[str, object]]] = []
    verification_started = monotonic()
    for index, day in enumerate(authorized, start=1):
        path, revision = _verified_source(manifest_root=manifest_root, day=day)
        pinned.append((day, path, revision))
        if index % progress_every == 0 or index == len(authorized):
            print(
                f"Verified {index}/{len(authorized)} source files "
                f"({monotonic() - verification_started:.1f}s elapsed)",
                flush=True,
            )
    requested = {*tickers, factor_ticker}
    sessions: dict[str, list[Session]] = {ticker: [] for ticker in requested}
    sources: list[dict[str, str]] = []
    parse_started = monotonic()
    cache_hits = 0
    for index, (day, path, revision) in enumerate(pinned, start=1):
        session = pd.Timestamp(day)
        opening = calendar.session_open(session).to_pydatetime()
        closing = calendar.session_close(session).to_pydatetime()
        ingested_at = datetime.now(UTC)
        if extract_cache_root is None:
            records = parse_selected_minutes(
                path,
                session=day,
                tickers=requested,
                opening=opening,
                closing=closing,
                publication_lag_seconds=publication_lag_seconds,
                ingested_at=ingested_at,
            )
        else:
            records, hit = load_or_parse_selected_minutes(
                path,
                cache_root=extract_cache_root,
                source_sha256=str(revision["sha256"]),
                session=day,
                tickers=requested,
                opening=opening,
                closing=closing,
                publication_lag_seconds=publication_lag_seconds,
                ingested_at=ingested_at,
                parser=parse_selected_minutes,
            )
            cache_hits += int(hit)
        for ticker in requested:
            if records[ticker]:
                sessions[ticker].append(_to_session(day, records[ticker]))
        sources.append({"day": day.isoformat(), "sha256": str(revision["sha256"])})
        if index % progress_every == 0 or index == len(pinned):
            print(
                f"Extracted {index}/{len(pinned)} sessions "
                f"(cache hits: {cache_hits}; {monotonic() - parse_started:.1f}s elapsed)",
                flush=True,
            )
    # Factor coverage is mandatory. Missing market days cannot be silently
    # removed when calculating security-specific strategy returns.
    factor_by_day = {s.day: s for s in sessions[factor_ticker]}
    missing_factor = set(authorized) - set(factor_by_day)
    if missing_factor:
        raise ValueError(f"missing {factor_ticker} minute data on {len(missing_factor)} dates")
    # Security-specific missing days are flagged; for a broad historical
    # universe, survivorship and delisting require a proper point-in-time
    # universe before interpreting pooled profitability.
    results: dict[str, Any] = {}
    for ticker in tickers:
        print(f"Scoring {ticker}...", flush=True)
        paired = [
            (asset, factor_by_day[asset.day])
            for asset in sessions[ticker]
            if asset.day in factor_by_day
        ]
        asset_days = [pair[0] for pair in paired]
        factor_days = [pair[1] for pair in paired]
        results[ticker] = evaluate_opening_range_history(
            asset_sessions=asset_days,
            factor_sessions=factor_days,
            validation_start=validation_start,
            horizons=horizons,
            costs_bps=costs_bps,
            bootstrap_replicates=bootstrap_replicates,
        )
        results[ticker]["source_days_missing_asset"] = sorted(
            day.isoformat() for day in authorized if day not in {s.day for s in asset_days}
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "holdout_accessed": False,
        "original_protected_days_skipped": sorted(
            day.isoformat() for day in protected if start <= day <= end
        ),
        "original_receipt_sha256": hashlib.sha256(original_receipt_path.read_bytes()).hexdigest(),
        "source_manifests": sources,
        "configuration": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "publication_lag_seconds_assumed_not_verified": publication_lag_seconds,
            "factor_ticker": factor_ticker,
            "horizons": list(horizons),
            "roundtrip_costs_bps": list(costs_bps),
        },
        "limitations": [
            "historical ticker strings are not point-in-time identity-verified",
            "unadjusted aggregate prices need corporate-action review",
            "minute open/close proxies do not represent executable quote fills",
            "multiple tickers may signal concurrently; results are not portfolio P&L",
            "data source availability time is assumed, not historically verified",
        ],
        "tickers": results,
        "extraction_cache_hits": cache_hits,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", required=True)
    parser.add_argument("--factor-ticker", default="QQQ")
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/minute_manifests")
    )
    parser.add_argument(
        "--original-receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--validation-start", required=True, type=date.fromisoformat)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument(
        "--extract-cache-root",
        type=Path,
        default=Path("data/dev/massive_flat_files/research_extracts"),
    )
    parser.add_argument("--no-extract-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--roundtrip-cost-bps", type=float, action="append")
    args = parser.parse_args()
    report = run_flat_history_research(
        tickers=tuple(t.upper() for t in args.ticker),
        factor_ticker=args.factor_ticker.upper(),
        manifest_root=resolve_project_path(args.manifest_root),
        original_receipt_path=resolve_project_path(args.original_receipt),
        start=args.start,
        end=args.end,
        validation_start=args.validation_start,
        output_dir=resolve_project_path(args.output_dir),
        horizons=tuple(args.horizon or (30, 60)),
        costs_bps=tuple(args.roundtrip_cost_bps or (4.0, 8.0, 12.0)),
        extract_cache_root=(
            None if args.no_extract_cache else resolve_project_path(args.extract_cache_root)
        ),
        progress_every=args.progress_every,
    )
    for ticker, result in report["tickers"].items():
        for experiment in result["experiments"]:
            horizon = experiment["horizon_minutes"]
            for variant, scores in experiment["variants"].items():
                for scenario in scores["cost_sensitivity"]:
                    print(
                        f"{ticker} {horizon}m {variant} "
                        f"cost={scenario['roundtrip_cost_bps']:g}bps "
                        f"trades={scenario['trades']} "
                        f"net/session={scenario['net_bps_per_session']:+.3f}bps"
                    )
    print("NOTICE: price proxies only; protected original sessions not loaded")


if __name__ == "__main__":
    main()
