"""P4.6: early-to-late ETF momentum using verified Massive minute flat files.

Research-only. Never reads protected original holdout files or touches a broker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.ingestion.minute_research_extract import (
    load_or_parse_selected_minutes,
    parse_selected_minutes_fast,
)
from quant_research.model.first_passage_validation import MinuteBar, Session
from quant_research.model.late_session_momentum import (
    SIGNALS,
    LateSessionEvent,
    build_late_session_event,
    score_late_session,
)
from quant_research.model.opening_range_history import protected_research_days
from quant_research.paths import resolve_project_path


def _verified_source(*, manifest_root: Path, day: date) -> tuple[Path, str]:
    """Require exact source revision; do not trust caches without verification."""
    path = manifest_root / f"{day}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz"
    revisions = payload.get("revisions")
    if (
        payload.get("format") != 1
        or payload.get("bucket") != "flatfiles"
        or payload.get("key") != key
        or not isinstance(revisions, list)
        or len(revisions) != 1
    ):
        raise ValueError(f"unexpected Massive manifest: {day}")
    revision = revisions[0]
    source = Path(revision["path"])
    if not source.is_file():
        raise FileNotFoundError(f"source file missing: {source}")
    sha = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != revision["sha256"] or source.stat().st_size != revision["bytes"]:
        raise ValueError(f"source checksum or size mismatch: {day}")
    return source, sha.hexdigest()


def _session(*, day: date, rows: list[dict[str, object]]) -> Session:
    bars = tuple(
        MinuteBar(
            timestamp=cast(datetime, row["timestamp"]),
            available_at=cast(datetime, row["available_at"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in rows
    )
    return Session(day, bars)


def run_late_session_research(
    *,
    manifest_root: Path,
    original_receipt_path: Path,
    start: date,
    end: date,
    validation_start: date,
    output_dir: Path,
    tickers: tuple[str, ...] = ("SPY", "QQQ"),
    costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0, 20.0),
    publication_lag_seconds: int = 60,
    bootstrap_replicates: int = 1000,
    progress_every: int = 10,
    extract_cache_root: Path = Path("data/dev/massive_flat_files/research_extracts"),
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"will not overwrite report: {output_dir}")
    if not start < validation_start <= end:
        raise ValueError("start < validation_start <= end required")
    if (
        not tickers
        or len(set(tickers)) != len(tickers)
        or any(not t.isascii() or not t.isupper() or not t.isalpha() for t in tickers)
        or not costs_bps
        or any(not isfinite(c) or c < 0 for c in costs_bps)
        or publication_lag_seconds < 1
        or progress_every < 1
        or bootstrap_replicates < 1
    ):
        raise ValueError("invalid research configuration")
    receipt = original_receipt_path.read_bytes()
    protected = protected_research_days(json.loads(receipt))
    cal = xcals.get_calendar("XNYS")
    all_days = tuple(
        t.date() for t in cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    )
    allowed = tuple(day for day in all_days if day not in protected)
    eligible = tuple(day for day in allowed if day >= validation_start)
    if len(allowed) < 3 or not eligible:
        raise ValueError("insufficient authorized research days")

    previous_close: dict[str, float | None] = {t: None for t in tickers}
    events: dict[str, list[LateSessionEvent]] = {t: [] for t in tickers}
    source_revisions: list[dict[str, str]] = []
    missing: dict[str, int] = {t: 0 for t in tickers}
    unavailable_prior: dict[str, int] = {t: 0 for t in tickers}
    cache_hits = 0
    start_time = monotonic()
    # Exclude protected dates BEFORE source verification or cache reading.
    for index, day in enumerate(all_days, start=1):
        if day in protected:
            previous_close = {t: None for t in tickers}
            continue
        source, sha = _verified_source(manifest_root=manifest_root, day=day)
        source_revisions.append({"day": day.isoformat(), "sha256": sha})
        stamp = pd.Timestamp(day)
        market_open = cal.session_open(stamp).to_pydatetime()
        market_close = cal.session_close(stamp).to_pydatetime()
        raw, hit = load_or_parse_selected_minutes(
            source,
            cache_root=extract_cache_root,
            source_sha256=sha,
            session=day,
            tickers=set(tickers),
            opening=market_open,
            closing=market_close,
            publication_lag_seconds=publication_lag_seconds,
            ingested_at=datetime.now(UTC),
            parser=parse_selected_minutes_fast,
        )
        cache_hits += int(hit)
        for ticker in tickers:
            session = _session(day=day, rows=raw[ticker])
            prior = previous_close[ticker]
            if day >= validation_start and prior is not None:
                event = build_late_session_event(
                    ticker=ticker,
                    session=session,
                    previous_close=prior,
                    opening=market_open,
                    closing=market_close,
                )
                if event is None:
                    missing[ticker] += 1
                else:
                    events[ticker].append(event)
            elif day >= validation_start:
                unavailable_prior[ticker] += 1
            # The previous close only becomes known AFTER that day completes.
            # Missing closing bars invalidate next day's overnight reference.
            last = [b for b in session.bars if b.timestamp == market_close]
            previous_close[ticker] = (
                last[0].close if len(last) == 1 and len(session.bars) == 390 else None
            )
        if index % progress_every == 0 or index == len(all_days):
            print(
                f"Processed {index}/{len(all_days)} sessions; "
                f"extraction cache hits={cache_hits}; elapsed={monotonic() - start_time:.1f}s",
                flush=True,
            )

    results: dict[str, object] = {}
    for ticker in tickers:
        results[ticker] = {
            "scored_events": len(events[ticker]),
            "missing_or_ineligible": missing[ticker],
            "no_usable_previous_close": unavailable_prior[ticker],
            "signals": {
                name: score_late_session(
                    events=events[ticker],
                    eligible_days=eligible,
                    signal=name,
                    costs_bps=costs_bps,
                    replications=bootstrap_replicates,
                )
                for name in SIGNALS
            },
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_family": "ETF-early-to-late-intraday-momentum",
        "research_only": True,
        "proxy_not_executable": True,
        "holdout_accessed": False,
        "original_protected_days_skipped": sorted(
            d.isoformat() for d in protected if start <= d <= end
        ),
        "original_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        "source_revisions": source_revisions,
        "configuration": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "scored_sessions": len(eligible),
            "tickers": list(tickers),
            "costs_bps": list(costs_bps),
            "opening_feature": "first-30m close / prior close or session open",
            "decision_time": "15:30 New York, standard sessions",
            "entry_proxy": "15:31 minute open",
            "exit_proxy": "15:59 minute open, before closing auction",
            "publication_lag_seconds_assumed": publication_lag_seconds,
        },
        "extract_cache_hits": cache_hits,
        "results": results,
        "limitations": [
            "Early closing sessions are excluded from event generation.",
            "Minute open proxies are not executable quotes and omit fill risk.",
            "Short borrowing, short-sale restrictions and funding are excluded.",
            "Flat-file historical ETF prices are not adjusted for dividends or splits.",
            "Previously inspected 2023-2024 dates are development only.",
            "Absolute return comparisons are not risk-adjusted or annualized.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--ticker", action="append", help="ETF ticker (default: SPY and QQQ)")
    parser.add_argument("--roundtrip-cost-bps", type=float, action="append")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run_late_session_research(
        manifest_root=resolve_project_path(args.manifest_root),
        original_receipt_path=resolve_project_path(args.original_receipt),
        start=args.start,
        end=args.end,
        validation_start=args.validation_start,
        output_dir=resolve_project_path(args.output_dir),
        tickers=tuple(args.ticker or ("SPY", "QQQ")),
        costs_bps=tuple(args.roundtrip_cost_bps or (4.0, 8.0, 12.0, 20.0)),
        bootstrap_replicates=args.bootstrap_replicates,
        progress_every=args.progress_every,
        extract_cache_root=resolve_project_path(
            Path("data/dev/massive_flat_files/research_extracts")
        ),
    )
    for ticker, result in report["results"].items():
        for signal, score in result["signals"].items():
            for row in score["cost_sensitivity"]:
                print(
                    f"{ticker} {signal}: trades={row['trades']} "
                    f"cost={row['roundtrip_cost_bps']:g}bps "
                    f"net/session={row['net_bps_per_session']:+.3f}bps "
                    f"CI={row['session_bootstrap_95ci']} [PROXY ONLY]"
                )
    print("NOTICE: development-only; original protected dates not loaded; no broker access")


if __name__ == "__main__":
    main()
