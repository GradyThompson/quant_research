"""P6: build source-pinned, historically classified intraday barrier observations.

Collect retrospective dated Massive reference/split evidence. Availability of
vendor records at historical decision time is NOT independently certified.
Never open protected dates or bridge their liquidity history.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from datetime import UTC, date, datetime
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import pandas as pd
from dotenv import load_dotenv

from quant_research.data.ingestion.minute_market_summary import (
    DailySummary,
    SummaryJob,
    summarize_market_job,
)
from quant_research.data.ingestion.minute_research_extract import (
    load_or_parse_selected_minutes,
)
from quant_research.model.ml_barrier import construct_barrier_row, row_to_json
from quant_research.model.opening_range_history import protected_research_days
from quant_research.paths import PROJECT_ROOT


class MassiveEvidence:
    """Immutable cached Massive REST responses; paginate and fail closed."""

    def __init__(self, key: str, root: Path) -> None:
        self.key = key
        self.root = root

    def query(self, family: str, parameters: dict[str, str]) -> list[dict[str, Any]]:
        canonical = family + "?" + urlencode(sorted(parameters.items()))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        path = self.root / family / f"{digest}.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("query") != canonical or not isinstance(saved.get("results"), list):
                raise ValueError(f"corrupt evidence cache: {path}")
            return saved["results"]
        if not self.key:
            raise ValueError("MASSIVE_API_KEY required to collect missing reference/split evidence")
        url = f"https://api.massive.com/{family}?{urlencode(parameters)}"
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        while url:
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.netloc != "api.massive.com" or url in seen:
                raise ValueError("invalid Massive pagination URL")
            seen.add(url)
            for attempt in range(5):
                req = Request(url, headers={"Authorization": f"Bearer {self.key}"})
                try:
                    with urlopen(req, timeout=45) as response:
                        payload = json.load(response)
                    break
                except HTTPError as exc:
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                        raise RuntimeError(f"Massive returned HTTP {exc.code}: {family}") from exc
                    time.sleep(min(2**attempt, 16))
                except URLError as exc:
                    if attempt == 4:
                        raise RuntimeError(f"Massive network error: {family}") from exc
                    time.sleep(min(2**attempt, 16))
            rows = payload.get("results")
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                raise ValueError("Massive response missing results array")
            found.extend(rows)
            next_url = payload.get("next_url")
            if next_url is not None and not isinstance(next_url, str):
                raise ValueError("invalid pagination field")
            url = next_url or ""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"query": canonical, "results": found}, handle, separators=(",", ":"))
        return found

    def common_stocks(self, day: date) -> set[str]:
        rows = self.query(
            "v3/reference/tickers",
            {
                "date": day.isoformat(),
                "market": "stocks",
                "locale": "us",
                "type": "CS",
                "active": "true",
                "limit": "1000",
            },
        )
        return {
            str(row["ticker"])
            for row in rows
            if row.get("type") == "CS"
            and row.get("locale") == "us"
            and row.get("share_class_figi")
            and row.get("ticker")
        }

    def split_days(self, start: date, end: date) -> set[tuple[date, str]]:
        rows = self.query(
            "stocks/v1/splits",
            {
                "execution_date.gte": start.isoformat(),
                "execution_date.lte": end.isoformat(),
                "limit": "5000",
            },
        )
        return {(date.fromisoformat(str(r["execution_date"])), str(r["ticker"])) for r in rows}


def verified_source(root: Path, day: date) -> tuple[Path, str]:
    """Verify pinned compressed bytes even when downstream summary is cached."""
    manifest = json.loads((root / f"{day}.json").read_text(encoding="utf-8"))
    expected = f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz"
    revisions = manifest.get("revisions")
    if (
        manifest.get("format") != 1
        or manifest.get("key") != expected
        or manifest.get("bucket") != "flatfiles"
        or not isinstance(revisions, list)
        or len(revisions) != 1
    ):
        raise ValueError(f"invalid source receipt for {day}")
    revision = revisions[0]
    path = Path(revision["path"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if sha != revision["sha256"] or path.stat().st_size != revision["bytes"]:
        raise ValueError(f"source revision mismatch for {day}")
    return path, sha


def choose_liquid(
    current: dict[str, DailySummary],
    historical: list[dict[str, DailySummary]],
    identities: set[str],
    split_today: set[str],
    *,
    universe: int,
    min_adv: float,
    min_price: float,
) -> list[tuple[str, float, float]]:
    """Only completed prior-day summary fields influence daily liquidity screen."""
    if len(historical) != 20:
        return []
    eligible: list[tuple[str, float, float]] = []
    for ticker, current_row in current.items():
        if (
            ticker not in identities
            or ticker in split_today
            or current_row.opening_open < min_price
        ):
            continue
        prior = [day.get(ticker) for day in historical]
        if any(row is None or not row.close_eligible for row in prior):
            continue
        filled = [row for row in prior if row is not None]
        adv = sum(row.dollar_volume for row in filled) / len(filled)
        opening_volume = sum(row.opening_volume for row in filled) / len(filled)
        if adv < min_adv or opening_volume <= 0:
            continue
        eligible.append((ticker, adv, opening_volume))
    eligible.sort(key=lambda r: (-r[1], r[0]))
    return eligible[:universe]


def _source_summary_task(
    day: date,
    manifest_root: Path,
    summary_root: Path,
    opening: datetime,
    closing: datetime,
) -> tuple[Path, str, dict[str, DailySummary]]:
    """Verify a raw file and obtain its cached market summary in a worker."""
    path, sha = verified_source(manifest_root, day)
    job: SummaryJob = (day, path, sha, summary_root, opening, closing)
    _, summary, _ = summarize_market_job(job)
    return path, sha, summary


def _day_cache_path(
    day: date,
    sha: str,
    history_shas: list[str],
    universe: int,
    split_fingerprint: str,
    day_cache_root: Path,
) -> tuple[Path, list[Any]]:
    # Keep this identity EXACTLY compatible with the original P6 day cache.
    identity: list[Any] = [
        "P6-day-v3",
        day.isoformat(),
        sha,
        history_shas,
        universe,
        split_fingerprint,
    ]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return day_cache_root / f"{day:%Y}" / f"{day}-{digest}.json.gz", identity


def _read_day_cache(path: Path, identity: list[Any]) -> tuple[list[dict[str, Any]], int, int]:
    with gzip.open(path, "rt", encoding="utf-8") as cached:
        payload = json.load(cached)
    if payload.get("identity") != identity or not isinstance(payload.get("rows"), list):
        raise ValueError(f"invalid P6 day cache: {path}")
    return (
        payload["rows"],
        int(payload.get("eligible_candidates", 0)),
        int(payload.get("missing_or_censored", 0)),
    )


def _label_day_task(
    day: date,
    path: Path,
    sha: str,
    opening: datetime,
    closing: datetime,
    extract_root: Path,
    cache_path: Path,
    identity: list[Any],
    selected: list[tuple[str, float, float]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Extract and label independent symbols; atomically publish the daily cache."""
    # A concurrent/resumed run may have committed the same immutable cache.
    if cache_path.is_file():
        return _read_day_cache(cache_path, identity)
    day_rows: list[dict[str, Any]] = []
    missing = 0
    if selected:
        tickers = {name for name, _, _ in selected} | {"SPY"}
        bars, _ = load_or_parse_selected_minutes(
            path,
            cache_root=extract_root,
            source_sha256=sha,
            session=day,
            tickers=tickers,
            opening=opening,
            closing=closing,
            publication_lag_seconds=60,
            ingested_at=datetime.now(UTC),
        )
        market = bars.get("SPY", [])
        if len(market) >= 90:
            for ticker, adv, opening_vol in selected:
                row = construct_barrier_row(
                    day=day,
                    ticker=ticker,
                    asset=bars.get(ticker, []),
                    market=market,
                    prior_adv=adv,
                    opening_volume_baseline=opening_vol,
                )
                if row is not None:
                    day_rows.append(row_to_json(row))
                else:
                    missing += 1
    payload = {
        "identity": identity,
        "rows": day_rows,
        "eligible_candidates": len(selected),
        "missing_or_censored": missing,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent,
        prefix="p6-day-",
        suffix=".tmp",
        delete=False,
    ) as temp:
        temporary = Path(temp.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return day_rows, len(selected), missing


def build_dataset(
    *,
    start: date,
    end: date,
    validation_start: date,
    receipt: Path,
    manifest_root: Path,
    output_dir: Path,
    evidence_root: Path,
    summary_root: Path,
    extract_root: Path,
    api_key: str,
    day_cache_root: Path | None = None,
    universe: int = 500,
    max_sessions: int = 0,
    progress_every: int = 10,
    workers: int = 1,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P6 dataset: {output_dir}")
    if (
        start >= validation_start
        or validation_start > end
        or universe < 1
        or progress_every < 1
        or workers < 1
    ):
        raise ValueError("invalid dataset dates or parameters")
    receipt_bytes = receipt.read_bytes()
    protected = protected_research_days(json.loads(receipt_bytes))
    cal = xcals.get_calendar("XNYS")
    days = [stamp.date() for stamp in cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]
    if max_sessions > 0:
        days = days[:max_sessions]
    provider = MassiveEvidence(api_key, evidence_root)
    splits = provider.split_days(start, end)
    split_fingerprint = hashlib.sha256(
        json.dumps(sorted((d.isoformat(), ticker) for d, ticker in splits)).encode()
    ).hexdigest()
    splits_by_day: dict[date, set[str]] = {}
    for split_day, ticker in splits:
        splits_by_day.setdefault(split_day, set()).add(ticker)
    history: deque[dict[str, DailySummary]] = deque(maxlen=20)
    history_shas: deque[str] = deque(maxlen=20)
    sources: list[dict[str, str]] = []
    started_at = time.perf_counter()
    time_wait_summary = 0.0
    time_reference = 0.0
    time_wait_labels = 0.0
    day_cache_hits = 0
    day_cache_misses = 0
    counts: dict[str, int] = {
        "rows": 0,
        "sessions": 0,
        "ambiguous_stops": 0,
        "eligible_candidates": 0,
        "missing_or_censored": 0,
    }
    if day_cache_root is None:
        day_cache_root = Path("data/dev/massive_flat_files/p6_barrier_day_cache")
    day_cache_root.mkdir(parents=True, exist_ok=True)
    tmp_rows = tempfile.NamedTemporaryFile(
        dir=day_cache_root,
        prefix="p6-build-",
        suffix=".jsonl.gz",
        delete=False,
    )
    tmp_path = Path(tmp_rows.name)
    tmp_rows.close()

    # Schedule *only* permitted sessions. Limit prefetch to avoid retaining
    # hundreds of market-wide summaries and overwhelming the local disk.
    permitted = [day for day in days if day not in protected]
    prepared: deque[tuple[date, Future[tuple[Path, str, dict[str, DailySummary]]]]] = deque()
    pending_labels: deque[tuple[date, Future[tuple[list[dict[str, Any]], int, int]]]] = deque()
    submitted = 0
    prefetch_limit = max(1, workers * 2)

    def next_summary(
        executor: ProcessPoolExecutor | None,
        day: date,
        opening: datetime,
        closing: datetime,
    ) -> tuple[Path, str, dict[str, DailySummary]]:
        nonlocal submitted
        if executor is None:
            return _source_summary_task(day, manifest_root, summary_root, opening, closing)
        while len(prepared) < prefetch_limit and submitted < len(permitted):
            candidate = permitted[submitted]
            stamp = pd.Timestamp(candidate)
            future = executor.submit(
                _source_summary_task,
                candidate,
                manifest_root,
                summary_root,
                cal.session_open(stamp).to_pydatetime(),
                cal.session_close(stamp).to_pydatetime(),
            )
            prepared.append((candidate, future))
            submitted += 1
        found_day, future = prepared.popleft()
        if found_day != day:
            raise RuntimeError("P6 prefetched source day order mismatch")
        return future.result()

    def flush_label(
        out: Any,
        future: Future[tuple[list[dict[str, Any]], int, int]],
        day: date,
    ) -> None:
        nonlocal time_wait_labels
        tick = time.perf_counter()
        day_rows, eligible, missing = future.result()
        time_wait_labels += time.perf_counter() - tick
        counts["eligible_candidates"] += eligible
        counts["missing_or_censored"] += missing
        counts["sessions"] += int(day >= validation_start)
        for record in day_rows:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            counts["rows"] += 1
            counts["ambiguous_stops"] += int(record["ambiguous"])

    # A single worker preserves exact old behavior, including monkeypatched
    # fixtures and deterministic serial debugging. Workers only receive paths
    # and small selected-symbol lists, never the 20-day summary history.
    # Python 3.13 warns when fork() is called from a multithreaded parent,
    # as can happen under pytest and in interactive research environments.
    # Use a fresh, importable worker process rather than inheriting threads.
    # forkserver is available on Linux/POSIX; spawn is the portable fallback.
    start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
    context = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context(start_method),
        )
        if workers > 1
        else nullcontext(None)
    )
    try:
        with context as executor, gzip.open(tmp_path, "wt", encoding="utf-8") as out:
            for index, day in enumerate(days, 1):
                if day in protected:
                    history.clear()
                    history_shas.clear()
                    continue
                stamp = pd.Timestamp(day)
                opening = cal.session_open(stamp).to_pydatetime()
                closing = cal.session_close(stamp).to_pydatetime()
                tick = time.perf_counter()
                path, sha, current = next_summary(executor, day, opening, closing)
                time_wait_summary += time.perf_counter() - tick
                sources.append({"day": day.isoformat(), "sha256": sha})
                if len(history) == 20:
                    cache_path, identity = _day_cache_path(
                        day,
                        sha,
                        list(history_shas),
                        universe,
                        split_fingerprint,
                        day_cache_root,
                    )
                    if cache_path.is_file():
                        day_cache_hits += 1
                        day_rows, eligible, missing = _read_day_cache(cache_path, identity)
                        completed: Future[tuple[list[dict[str, Any]], int, int]] = Future()
                        completed.set_result((day_rows, eligible, missing))
                        future = completed
                    else:
                        day_cache_misses += 1
                        tick = time.perf_counter()
                        identities = provider.common_stocks(day)
                        time_reference += time.perf_counter() - tick
                        selected = choose_liquid(
                            current,
                            list(history),
                            identities,
                            splits_by_day.get(day, set()),
                            universe=universe,
                            min_adv=50_000_000.0,
                            min_price=20.0,
                        )
                        arguments = (
                            day,
                            path,
                            sha,
                            opening,
                            closing,
                            extract_root,
                            cache_path,
                            identity,
                            selected,
                        )
                        if executor is None:
                            completed = Future()
                            completed.set_result(_label_day_task(*arguments))
                            future = completed
                        else:
                            future = executor.submit(_label_day_task, *arguments)
                    pending_labels.append((day, future))
                    if len(pending_labels) >= prefetch_limit:
                        past_day, past_future = pending_labels.popleft()
                        flush_label(out, past_future, past_day)
                # A split on this ex-date must not pollute the next 20
                # sessions' raw-price liquidity history.
                for ticker in splits_by_day.get(day, set()):
                    current.pop(ticker, None)
                history.append(current)
                history_shas.append(sha)
                if index % progress_every == 0 or index == len(days):
                    print(
                        f"P6 dataset {index}/{len(days)} days, "
                        f"{counts['rows']} written rows, "
                        f"{len(pending_labels)} pending days",
                        flush=True,
                    )
            while pending_labels:
                past_day, past_future = pending_labels.popleft()
                flush_label(out, past_future, past_day)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    if counts["rows"] == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError("no eligible P6 barrier rows; inspect reference and minute coverage")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "rows.jsonl.gz"
    shutil.move(str(tmp_path), rows_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "P6-ML-barrier-dataset",
        "validation_start": validation_start.isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "original_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "protected_dates": sorted(d.isoformat() for d in protected if start <= d <= end),
        "protected_holdout_opened": False,
        "source_revisions": sources,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "counts": counts,
        "max_universe": universe,
        "build_performance": {
            "workers": workers,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "summary_wait_seconds": round(time_wait_summary, 3),
            "reference_seconds": round(time_reference, 3),
            "label_wait_seconds": round(time_wait_labels, 3),
            "day_cache_hits": day_cache_hits,
            "day_cache_misses": day_cache_misses,
        },
        "baseline": "zero-price-drift mathematical first-passage using morning realized variance; not a refitted M2.5 seasonal forecast",
        "identity": "retrospective dated Massive CS+FIGI; original historical knowledge timestamps not verified",
        "actions": "split ex-dates excluded before signal; intraday same-session labels only",
        "limitations": [
            "Retrospective vendor historical identity is not independently proven known-at-decision.",
            "Missing/halts/censored securities are omitted and counted; do not interpret as full-universe economics.",
            "Price proxies without bid/ask execution and no full market-wide dividend events.",
            "P6 phase 1 is not a deployable portfolio backtest.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {rows_path}, {counts['rows']} observations", flush=True)
    print(
        f"P6 build: {time.perf_counter() - started_at:.1f}s elapsed; "
        f"{time_wait_summary:.1f}s summary wait, {time_reference:.1f}s reference, "
        f"{time_wait_labels:.1f}s label wait; "
        f"{day_cache_hits} cached days, {day_cache_misses} new days",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--validation-start", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--original-receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("data/dev/massive_flat_files/minute_manifests")
    )
    parser.add_argument(
        "--summary-root", type=Path, default=Path("data/dev/massive_flat_files/market_summaries")
    )
    parser.add_argument(
        "--extract-root", type=Path, default=Path("data/dev/massive_flat_files/research_extracts")
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("data/dev/massive_rest_evidence/p6")
    )
    parser.add_argument("--max-universe", type=int, default=500)
    parser.add_argument(
        "--day-cache-root",
        type=Path,
        default=Path("data/dev/massive_flat_files/p6_barrier_day_cache"),
    )
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel source verification, summaries and selected minute scans (try 2)",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    build_dataset(
        start=args.start,
        end=args.end,
        validation_start=args.validation_start,
        receipt=args.original_receipt,
        manifest_root=args.manifest_root,
        output_dir=args.output_dir,
        evidence_root=args.evidence_root,
        summary_root=args.summary_root,
        extract_root=args.extract_root,
        api_key=os.environ.get("MASSIVE_API_KEY", ""),
        day_cache_root=args.day_cache_root,
        universe=args.max_universe,
        max_sessions=args.max_sessions,
        progress_every=args.progress_every,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
