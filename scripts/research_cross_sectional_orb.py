"""P4.5: research-only, point-in-time market-wide abnormal-volume event study.

Uses locally cached Massive minute flat files. Original protected dates are
excluded BEFORE manifest verification or source loading. No broker access.
Historical instrument *types* are NOT identifiable from minute OHLCV alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, date, datetime
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from quant_research.data.ingestion.minute_market_summary import (
    SummaryJob,
    SummaryResult,
    summarize_market_job,
)
from quant_research.data.ingestion.minute_research_extract import (
    load_or_parse_selected_minutes,
    parse_selected_minutes_fast,
)
from quant_research.model.cross_sectional_orb import (
    Event,
    RollingUniverse,
    first_breakout_event,
    score_events,
)
from quant_research.model.first_passage_validation import MinuteBar, Session
from quant_research.model.opening_range_history import protected_research_days
from quant_research.paths import resolve_project_path


def _verified_source(*, manifest_root: Path, day: date) -> tuple[Path, str]:
    """Fail closed if the source file differs from its immutable manifest."""
    manifest_path = manifest_root / f"{day}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz"
    revisions = payload.get("revisions")
    if (
        payload.get("format") != 1
        or payload.get("bucket") != "flatfiles"
        or payload.get("key") != key
        or not isinstance(revisions, list)
        or len(revisions) != 1
    ):
        raise ValueError(f"source manifest mismatch: {day}")
    revision = revisions[0]
    path = Path(revision["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if sha != revision["sha256"] or path.stat().st_size != revision["bytes"]:
        raise ValueError(f"source hash/size mismatch: {day}")
    return path, sha


def _ordered_summaries(
    jobs: Sequence[SummaryJob],
    *,
    workers: int,
) -> Generator[SummaryResult, None, None]:
    """Bounded parallel source summarization, consumed in session order.

    Summary calculations are independent, while RollingUniverse.observe is
    strictly sequential. At most 2*workers daily results are in flight, rather
    than scheduling hundreds of large market summaries simultaneously.
    """
    if workers < 1:
        raise ValueError("summary workers must be positive")
    if workers == 1:
        for job in jobs:
            yield summarize_market_job(job)
        return

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        pending: deque[Future[SummaryResult]] = deque()
        iterator = iter(jobs)
        for _ in range(min(2 * workers, len(jobs))):
            pending.append(executor.submit(summarize_market_job, next(iterator)))
        while pending:
            yield pending.popleft().result()
            next_job = next(iterator, None)
            if next_job is not None:
                pending.append(executor.submit(summarize_market_job, next_job))


def _to_session(day: date, rows: list[dict[str, object]]) -> Session:
    return Session(
        day,
        tuple(
            MinuteBar(
                timestamp=row["timestamp"],
                available_at=row["available_at"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in rows
        ),
    )


def run_market_research(
    *,
    manifest_root: Path,
    original_receipt_path: Path,
    start: date,
    end: date,
    validation_start: date,
    output_dir: Path,
    horizons: tuple[int, ...] = (30, 60),
    costs_bps: tuple[float, ...] = (8.0, 12.0, 20.0),
    opening_minutes: int = 15,
    scan_minutes: int = 120,
    lookback_sessions: int = 20,
    top_n: int = 20,
    positions: int = 3,
    min_rvol: float = 1.5,
    min_dollar_volume: float = 20_000_000.0,
    factor_ticker: str = "QQQ",
    publication_lag_seconds: int = 60,
    bootstrap_replicates: int = 2000,
    progress_every: int = 10,
    summary_workers: int = 1,
    summary_cache_root: Path = Path("data/dev/massive_flat_files/market_summaries"),
    extract_cache_root: Path = Path("data/dev/massive_flat_files/research_extracts"),
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing report overwrite: {output_dir}")
    if not start < validation_start <= end:
        raise ValueError("require start < validation_start <= end")
    if (
        not horizons
        or any(h < 1 for h in horizons)
        or not costs_bps
        or progress_every < 1
        or summary_workers < 1
        or publication_lag_seconds < 1
        or opening_minutes != 15
        or scan_minutes <= opening_minutes
    ):
        raise ValueError("invalid horizons, timing, or execution assumptions")
    if not factor_ticker.isascii() or not factor_ticker.isupper():
        raise ValueError("invalid factor")
    receipt = original_receipt_path.read_bytes()
    protected = protected_research_days(json.loads(receipt))
    calendar = xcals.get_calendar("XNYS")
    all_days = tuple(
        item.date() for item in calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    )
    authorized = [day for day in all_days if day not in protected]
    validation_days = [day for day in authorized if day >= validation_start]
    if not authorized or not validation_days:
        raise ValueError("no permissible scoring sessions")
    verified: dict[date, tuple[Path, str]] = {}
    started = monotonic()
    for i, day in enumerate(authorized, start=1):
        verified[day] = _verified_source(manifest_root=manifest_root, day=day)
        if i % progress_every == 0 or i == len(authorized):
            print(
                f"Verified {i}/{len(authorized)} files ({monotonic() - started:.1f}s)", flush=True
            )

    selector = RollingUniverse(
        lookback=lookback_sessions,
        min_rvol=min_rvol,
        count=top_n,
        factor_ticker=factor_ticker,
        min_dollar_volume=min_dollar_volume,
    )
    collected: dict[int, list[Event]] = {h: [] for h in horizons}
    candidates_per_day: dict[date, dict[str, int]] = {}
    summary_hits = 0
    extract_hits = 0
    # Construct jobs ONLY for verified and authorized sessions. All futures
    # finish in source-date order before feeding the chronological selector.
    jobs: list[SummaryJob] = []
    for day in authorized:
        path, source_hash = verified[day]
        session = pd.Timestamp(day)
        jobs.append(
            (
                day,
                path,
                source_hash,
                summary_cache_root,
                calendar.session_open(session).to_pydatetime(),
                calendar.session_close(session).to_pydatetime(),
            )
        )
    summary_stream = _ordered_summaries(jobs, workers=summary_workers)
    try:
        for i, day in enumerate(all_days, start=1):
            if day in protected:
                selector.observe({})  # preserve historical gap without reading it
                continue
            returned_day, summaries, hit = next(summary_stream)
            if returned_day != day:
                raise ValueError("parallel summaries arrived out of chronological order")
            path, source_hash = verified[day]
            session = pd.Timestamp(day)
            opening = calendar.session_open(session).to_pydatetime()
            closing = calendar.session_close(session).to_pydatetime()
            summary_hits += int(hit)
            if factor_ticker not in summaries:
                raise ValueError(f"missing complete {factor_ticker} bars: {day}")
            selected = selector.observe(summaries)
            candidates_per_day[day] = {
                "high_rvol": sum(c.cohort == "high_rvol" for c in selected),
                "normal_rvol_control": sum(c.cohort == "normal_rvol_control" for c in selected),
            }
            if day >= validation_start and selected:
                tickers = {c.ticker for c in selected} | {factor_ticker}
                raw, hit = load_or_parse_selected_minutes(
                    path,
                    cache_root=extract_cache_root,
                    source_sha256=source_hash,
                    session=day,
                    tickers=tickers,
                    opening=opening,
                    closing=closing,
                    publication_lag_seconds=publication_lag_seconds,
                    ingested_at=datetime.now(UTC),
                    parser=parse_selected_minutes_fast,
                )
                extract_hits += int(hit)
                factor = _to_session(day, raw[factor_ticker])
                for c in selected:
                    asset = _to_session(day, raw[c.ticker])
                    # Single predeclared event per (day, ticker, horizon), with
                    # no variant-specific search for a later convenient breakout.
                    for h in horizons:
                        event = first_breakout_event(
                            candidate=c,
                            asset=asset,
                            factor=factor,
                            holding_minutes=h,
                            opening_minutes=opening_minutes,
                            scan_minutes=scan_minutes,
                        )
                        if event is not None:
                            collected[h].append(event)
            if i % progress_every == 0 or i == len(all_days):
                print(
                    f"Processed {i}/{len(all_days)} sessions "
                    f"(summary cache hits={summary_hits}; extraction hits={extract_hits})",
                    flush=True,
                )
    finally:
        summary_stream.close()
    results: dict[str, object] = {}
    for horizon, events in collected.items():
        results[str(horizon)] = {
            "events": len(events),
            "cohorts": {
                f"{cohort}_{'market' if confirmed else 'all'}": score_events(
                    events=events,
                    eligible_days=validation_days,
                    cohort=cohort,
                    confirmed=confirmed,
                    positions=positions,
                    costs_bps=costs_bps,
                    bootstrap_replicates=bootstrap_replicates,
                )
                for cohort in ("high_rvol", "normal_rvol_control")
                for confirmed in (False, True)
            },
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_family": "cross-sectional-abnormal-volume-opening-range",
        "research_only": True,
        "proxy_not_executable": True,
        "holdout_accessed": False,
        "original_protected_days_skipped": sorted(
            day.isoformat() for day in protected if start <= day <= end
        ),
        "original_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        "source_revisions": [
            {"day": day.isoformat(), "sha256": verified[day][1]} for day in authorized
        ],
        "configuration": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "scored_sessions": len(validation_days),
            "horizons": list(horizons),
            "costs_bps": list(costs_bps),
            "lookback_sessions": lookback_sessions,
            "top_n": top_n,
            "max_positions": positions,
            "min_rvol": min_rvol,
            "min_dollar_volume": min_dollar_volume,
            "opening_minutes": opening_minutes,
            "scan_minutes": scan_minutes,
            "breakout_buffer_bps": 5.0,
            "publication_lag_seconds_assumed": publication_lag_seconds,
            "summary_workers": summary_workers,
        },
        "candidate_counts": {
            day.isoformat(): values
            for day, values in candidates_per_day.items()
            if day >= validation_start
        },
        "summary_cache_hits": summary_hits,
        "extract_cache_hits": extract_hits,
        "results": results,
        "limitations": [
            "Historical symbol identity and security type are NOT verified. ETFs/ADRs can enter the screen.",
            "No corporate-action adjusted prices; extreme overnight gaps >30% excluded.",
            "Prior daily dollar volume is a liquidity proxy, not spread or executable depth.",
            "Minute open and close are not executable bid/ask fills; short borrow omitted.",
            "Equal capital slots held in cash when inactive, not broker account equity P&L.",
            "2023-2024 are previously inspected development dates, not independent holdout.",
            "Source minute publication lag is an assumption, not verified historical timing.",
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", action="append", type=int)
    parser.add_argument("--roundtrip-cost-bps", action="append", type=float)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--summary-workers", type=int, default=1)
    args = parser.parse_args()
    report = run_market_research(
        manifest_root=resolve_project_path(args.manifest_root),
        original_receipt_path=resolve_project_path(args.original_receipt),
        start=args.start,
        end=args.end,
        validation_start=args.validation_start,
        output_dir=resolve_project_path(args.output_dir),
        horizons=tuple(args.horizon or (30, 60)),
        costs_bps=tuple(args.roundtrip_cost_bps or (8.0, 12.0, 20.0)),
        top_n=args.top_n,
        positions=args.positions,
        progress_every=args.progress_every,
        summary_workers=args.summary_workers,
        summary_cache_root=resolve_project_path(
            Path("data/dev/massive_flat_files/market_summaries")
        ),
        extract_cache_root=resolve_project_path(
            Path("data/dev/massive_flat_files/research_extracts")
        ),
    )
    for horizon, result in report["results"].items():
        for name, cohort in result["cohorts"].items():
            for row in cohort["cost_sensitivity"]:
                print(
                    f"{horizon}m {name}: trades={row['trades']} "
                    f"cost={row['roundtrip_cost_bps']:g}bps "
                    f"net/session={row['net_bps_per_session']:+.3f}bps "
                    "[PROXY ONLY]"
                )
    print("NOTICE: development-only; protected sessions never opened; no broker access")


if __name__ == "__main__":
    main()
